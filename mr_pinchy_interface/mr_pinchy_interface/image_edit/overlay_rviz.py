#!/usr/bin/env python3
"""
overlay_calib.py

Publish a *still image* as a ROS2 image topic (+ CameraInfo), publish a
trajectory marker + a few safety spheres, and broadcast a *tunable* camera TF
so you can overlay the trajectory on the still in RViz2's Camera display and
eyeball-calibrate the camera pose.

The background is the ghost-trail composite produced by image_edit/video_overlay.py
(overlay_still.png). The safety spheres are picked from the bag by bag-relative
time (SPHERE_TIMES, seconds after the first robot_marker), so they land on the
robot ghosts in the still.

You tune the 6-DOF pose live, most naturally by grabbing the 6-DOF *interactive
marker* in RViz (red arrow = camera viewing direction; drag the rings to rotate,
the arrows to translate) and watching the overlay snap into place. The marker
feeds the same params, so you can still nudge them with rqt_reconfigure sliders or:
    ros2 param set /overlay_calib pitch -0.15
Both stay in sync. On mouse-release the pose is printed for pasting back into INIT.

Add an "InteractiveMarkers" display in RViz on topic /overlay_calib/update to
see the handle.

Frames:
    WORLD_FRAME ---(tunable x,y,z,roll,pitch,yaw)---> LINK_FRAME ---(fixed)---> OPTICAL_FRAME
You tune the *intuitive* x-forward LINK_FRAME; the fixed optical rotation is
applied for you, so the image projects correctly.

Run:
    python3 overlay_calib.py
(needs a sourced ROS2 environment with cv_bridge + tf_transformations)
"""

import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from rclpy.parameter import Parameter

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import (Marker, InteractiveMarker,
                                    InteractiveMarkerControl,
                                    InteractiveMarkerFeedback)
from geometry_msgs.msg import TransformStamped, Point, Pose
from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from interactive_markers import InteractiveMarkerServer
from cv_bridge import CvBridge

try:
    from tf_transformations import quaternion_from_euler
except ImportError:                                   # minimal fallback
    def quaternion_from_euler(roll, pitch, yaw):
        cy, sy = math.cos(yaw * .5), math.sin(yaw * .5)
        cp, sp = math.cos(pitch * .5), math.sin(pitch * .5)
        cr, sr = math.cos(roll * .5), math.sin(roll * .5)
        return [sr*cp*cy - cr*sp*sy,
                cr*sp*cy + sr*cp*sy,
                cr*cp*sy - sr*sp*cy,
                cr*cp*cy + sr*sp*sy]      # x, y, z, w


# --- rotation helpers (sxyz / extrinsic XYZ == ROS default, matches the
#     quaternion_from_euler above) used to convert between frame conventions ---
def rot_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def euler_from_rot(R):
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(R[2, 0]) < 0.999999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw  = math.atan2(R[1, 0], R[0, 0])
    else:                                             # gimbal lock
        roll = 0.0
        yaw  = math.atan2(-R[0, 1], R[1, 1])
    return roll, pitch, yaw

def quat_to_rot(x, y, z, w):
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix, consistent with
    rot_from_euler/euler_from_rot above (so a marker pose round-trips to the
    same roll/pitch/yaw the sliders use)."""
    n = math.sqrt(x*x + y*y + z*z + w*w) or 1.0
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


# ============================ CONFIG — edit these ============================
# Background still: the ghost-trail composite from image_edit/video_overlay.py.
STILL_IMAGE   = "/home/shaohang/workspace/underwater_ws/src/mr_pinchy/mr_pinchy_interface/mr_pinchy_interface/image_edit/overlay_still.png"

# The trajectory is the path of mr_pinchy/base_link_ekf, read live from this bag.
# /mocap/mr_pinchy/odom_ekf carries that pose (header.frame_id=mocap,
# child_frame_id=mr_pinchy/base_link_ekf). The points are published in `mocap`,
# but we reproject them into WORLD_FRAME using the bag's TF tree.
ROSBAG_PATH   = "/home/shaohang/workspace/underwater_ws/data_rosbag/cut.mcap"
ODOM_TOPIC    = "/mocap/mr_pinchy/odom_ekf"
TRAJ_STRIDE   = 10                       # keep every Nth odom sample (downsample the line)

# Show one safety sphere per timestamp, in seconds *after the first*
# /safety_filter/robot_marker message (bag-relative). These match the robot
# ghosts in overlay_still.png (video_overlay.py FRAME_TIMES).
SPHERE_TIMES  = [0.0, 33.0]

WORLD_FRAME   = "mocap"                  # frame your trajectory + camera live in
LINK_FRAME    = "camera_link"            # intuitive x-forward camera body frame (you tune THIS)
OPTICAL_FRAME = "camera_optical_frame"   # z-forward optical frame (image is published in this)

# Intrinsics from the QTM calibration, camera 9 (Miqus Video, serial 33253).
# QTM stores focal length / principal point in sensor units; divide by the
# sensor extent and multiply by the image size to get pixels (see tick()).
QTM_INTRINSIC = dict(fu=95175.140625, fv=95147.500000,     # focal length (sensor units)
                     cu=61906.769531, cv=31561.828125,     # principal point (sensor units)
                     su=122880.0,     sv=69632.0)           # sensor extent U/V (SensorMax+1)
# OpenCV plumb_bob [k1, k2, p1, p2, k3] from QTM (Radial1,2, Tang1,2, Radial3).
DIST = [0.388242, 0.185869, 0.000633, 0.003709, 0.0]

# Burn the overlay into the published image with OpenCV (applies the real lens
# distortion, which RViz's pinhole Camera display cannot). Set False to fall
# back to the marker-only overlay.
DRAW_OVERLAY = True
TRAJ_COLOR   = (0, 255, 0)               # BGR
TRAJ_THICK   = 2
# Safety spheres are drawn as shaded translucent "bubbles" (see _draw_sphere):
# a tinted Lambert-shaded ball with a bright rim (Fresnel) and a specular
# highlight, so they read as 3D glass instead of a flat disc.
SPHERE_COLOR = (255, 153, 51)            # BGR base tint of the bubble
SPHERE_ALPHA = 0.55                      # peak opacity (at the lit rim / highlight)

# Fallback trajectory (in WORLD_FRAME, meters) used only if the bag can't be read.
TRAJECTORY = [
    (0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (1.0, 0.1, 0.0),
    (1.5, 0.3, 0.0),
    (2.0, 0.6, 0.0),
]


def load_trajectory_from_bag(bag_path, topic, stride=1, target_frame=None):
    """Read the base_link_ekf path from an odom topic in a rosbag2 (mcap).

    Points are reprojected from the odom's header frame into `target_frame`
    using the TF tree stored in the same bag (/tf + /tf_static).
    """
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from tf2_msgs.msg import TFMessage
    from geometry_msgs.msg import PointStamped
    from tf2_geometry_msgs import do_transform_point
    import tf2_ros
    import rclpy.time

    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag_path, storage_id="mcap"),
                ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in types:
        raise KeyError(f"{topic} not in bag (have: {sorted(types)})")
    msg_type = get_message(types[topic])

    buf = tf2_ros.Buffer()
    raw, i = [], 0          # collect (header, position); TF may arrive interleaved
    while reader.has_next():
        tname, data, _ = reader.read_next()
        if tname in ("/tf", "/tf_static"):
            for tr in deserialize_message(data, TFMessage).transforms:
                if tname == "/tf_static":
                    buf.set_transform_static(tr, "bag")
                else:
                    buf.set_transform(tr, "bag")
        elif tname == topic:
            if i % stride == 0:
                m = deserialize_message(data, msg_type)
                raw.append((m.header, m.pose.pose.position))
            i += 1

    pts = []
    for header, p in raw:
        if not target_frame or target_frame == header.frame_id:
            pts.append((p.x, p.y, p.z))
            continue
        ps = PointStamped()
        ps.header = header
        ps.point.x, ps.point.y, ps.point.z = p.x, p.y, p.z
        tf = buf.lookup_transform(target_frame, header.frame_id, rclpy.time.Time())
        out = do_transform_point(ps, tf).point
        pts.append((out.x, out.y, out.z))
    return pts


def load_robot_markers_from_bag(bag_path, topic, times, target_frame=None):
    """Pick one /safety_filter/robot_marker sphere per requested time and return
    a single SPHERE_LIST marker with those centers reprojected into target_frame.

    `times` are seconds *after the first* robot_marker message (bag-relative);
    for each we keep the sphere whose header stamp is closest to it, so the
    spheres land on the robot ghosts in overlay_still.png.
    """
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from tf2_msgs.msg import TFMessage
    from geometry_msgs.msg import PointStamped
    from tf2_geometry_msgs import do_transform_point
    import tf2_ros
    import rclpy.time

    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag_path, storage_id="mcap"),
                ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in types:
        return None
    msg_type = get_message(types[topic])

    buf = tf2_ros.Buffer()
    samples, template = [], None       # (stamp_sec, header, position) per message
    while reader.has_next():
        tname, data, _ = reader.read_next()
        if tname in ("/tf", "/tf_static"):
            for tr in deserialize_message(data, TFMessage).transforms:
                if tname == "/tf_static":
                    buf.set_transform_static(tr, "bag")
                else:
                    buf.set_transform(tr, "bag")
        elif tname == topic:
            m = deserialize_message(data, msg_type)
            template = template or m
            stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            samples.append((stamp, m.header, m.pose.position))

    if template is None or not samples:
        return None

    # nearest sphere to each requested bag-relative time (dedup, keep order)
    t0 = samples[0][0]
    chosen, seen = [], set()
    for t in times:
        k = min(range(len(samples)), key=lambda j: abs(samples[j][0] - t0 - t))
        if k not in seen:
            seen.add(k)
            chosen.append(samples[k])

    out = Marker()
    out.header.frame_id = target_frame or template.header.frame_id
    out.ns, out.id = "robot", 0
    out.type = Marker.SPHERE_LIST
    out.action = Marker.ADD
    out.scale = template.scale          # per-sphere diameter
    out.color = template.color
    out.pose.orientation.w = 1.0
    for stamp, header, p in chosen:
        if not target_frame or target_frame == header.frame_id:
            out.points.append(Point(x=p.x, y=p.y, z=p.z))
            continue
        ps = PointStamped()
        ps.header = header
        ps.point = p
        tf = buf.lookup_transform(target_frame, header.frame_id, rclpy.time.Time())
        q = do_transform_point(ps, tf).point
        out.points.append(Point(x=q.x, y=q.y, z=q.z))
    return out

PLAY_FPS     = 15.0      # republish rate (keeps RViz fed as you tune the pose)

# Fixed link(x-fwd,y-left,z-up) -> optical(z-fwd,x-right,y-down) rotation.
OPTICAL_RPY = (-math.pi / 2, 0.0, -math.pi / 2)

# Initial guess for the camera pose, in WORLD_FRAME (mocap).
#
# Taken straight from the Qualisys (QTM) calibration: camera 9 (Miqus Video,
# serial 33253) in 20260521_100718.json -- the camera that shot output.avi.
# QTM gives the camera pose in the mocap frame, but in a camera convention whose
# OPTICAL (viewing) axis is the camera's local +Y. We map that into the ROS
# optical convention (Z-forward), then strip the fixed link->optical rotation
# (the node re-applies it) to land on the x-forward LINK pose the sliders tune.
QTM_R = np.array([[ 0.035435, -0.998466, -0.042553],
                  [-0.286646,  0.030636, -0.957547],
                  [ 0.957381,  0.046128, -0.285120]])

QTM_T_MM = (8445.603516, 139.904236, -9.500122)            # camera position, mm
# QTM camera axes -> ROS optical axes:  X_opt=Z_qtm, Y_opt=X_qtm, Z_opt=Y_qtm
# (the camera is rolled 180 deg about its viewing axis vs. the naive mapping, so
# right/down flip while the Z viewing axis stays the same -- this rotates the
# projected overlay 180 deg about the image center without touching the video.)
QTM_TO_OPTICAL = np.array([[ 0,  1,  0],
                           [ 0,  0,  1],
                           [ 1,  0,  0]])

_R_link = (QTM_R @ QTM_TO_OPTICAL) @ rot_from_euler(*OPTICAL_RPY).T
_roll, _pitch, _yaw = euler_from_rot(_R_link)
#  roll = 3.0, pitch = -0.28, yaw = -3.1 ( empirical eyeball refinement on top of the QTM calibration (rqt tuning))
INIT = dict(x=QTM_T_MM[0] / 1000.0, y=QTM_T_MM[1] / 1000.0, z=QTM_T_MM[2] / 1000.0,
            roll=_roll, pitch=_pitch, yaw=_yaw)
# ===========================================================================


class OverlayCalib(Node):
    def __init__(self):
        super().__init__("overlay_calib")

        # tunable pose params, with ranges so rqt_reconfigure shows sliders
        self._dparam("x", INIT["x"], -10.0, 10.0)
        self._dparam("y", INIT["y"], -10.0, 10.0)
        self._dparam("z", INIT["z"], -10.0, 10.0)
        self._dparam("roll",  INIT["roll"],  -math.pi, math.pi)
        self._dparam("pitch", INIT["pitch"], -math.pi, math.pi)
        self._dparam("yaw",   INIT["yaw"],   -math.pi, math.pi)

        self.bridge   = CvBridge()
        self.img_pub  = self.create_publisher(Image, "camera/image_raw", 10)
        self.info_pub = self.create_publisher(CameraInfo, "camera/camera_info", 10)
        self.mark_pub = self.create_publisher(Marker, "trajectory", 10)
        self.robot_pub = self.create_publisher(Marker, "robot_marker", 10)
        self.robot_marker = None
        self.tf_bc        = TransformBroadcaster(self)
        self.static_bc    = StaticTransformBroadcaster(self)

        self._publish_optical_static()    # fixed LINK -> OPTICAL

        # trajectory = path of mr_pinchy/base_link_ekf, read from the bag
        self.trajectory = TRAJECTORY
        try:
            pts = load_trajectory_from_bag(ROSBAG_PATH, ODOM_TOPIC, TRAJ_STRIDE,
                                           target_frame=WORLD_FRAME)
            if pts:
                self.trajectory = pts
                self.get_logger().info(
                    f"Loaded {len(pts)} trajectory points from {ODOM_TOPIC}")
        except Exception as e:
            self.get_logger().warn(
                f"Falling back to hardcoded trajectory ({e})")

        # safety_filter spheres at SPHERE_TIMES, reprojected into WORLD_FRAME
        try:
            self.robot_marker = load_robot_markers_from_bag(
                ROSBAG_PATH, "/safety_filter/robot_marker", SPHERE_TIMES,
                target_frame=WORLD_FRAME)
            if self.robot_marker is not None:
                self.get_logger().info(
                    f"Loaded {len(self.robot_marker.points)} robot_marker spheres "
                    f"at t={SPHERE_TIMES}s from bag")
        except Exception as e:
            self.get_logger().warn(f"Could not load robot_marker ({e})")

        # numpy caches for the OpenCV overlay projection
        self.traj_np = np.array(self.trajectory, dtype=np.float64) \
            if self.trajectory else None
        if self.robot_marker is not None and self.robot_marker.points:
            self.sphere_np = np.array(
                [[p.x, p.y, p.z] for p in self.robot_marker.points], dtype=np.float64)
            self.sphere_radius = self.robot_marker.scale.x / 2.0
        else:
            self.sphere_np, self.sphere_radius = None, 0.0

        # static background: the ghost-trail composite still
        self.still = cv2.imread(STILL_IMAGE, cv2.IMREAD_COLOR)
        if self.still is None:
            self.get_logger().error(f"Could not read still image: {STILL_IMAGE}")

        # 6-DOF interactive marker: grab/drag the camera pose with the mouse in
        # RViz instead of nudging rqt sliders. Feedback flows back into the same
        # x/y/z/roll/pitch/yaw params, so the accurate OpenCV overlay updates live.
        self._setup_interactive_marker()

        self.timer = self.create_timer(1.0 / PLAY_FPS, self.tick)
        self.get_logger().info(
            "overlay_calib running. Drag the camera marker in RViz "
            "(or tune with rqt_reconfigure).")

    def _dparam(self, name, default, lo, hi):
        self.declare_parameter(
            name, default,
            ParameterDescriptor(floating_point_range=[FloatingPointRange(from_value=lo, to_value=hi)]))

    def _publish_optical_static(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = LINK_FRAME
        t.child_frame_id  = OPTICAL_FRAME
        # standard link(x-fwd,y-left,z-up) -> optical(z-fwd,x-right,y-down)
        q = quaternion_from_euler(*OPTICAL_RPY)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        self.static_bc.sendTransform(t)

    # ----- interactive-marker pose control (mouse-drag calibration) ----------
    def _param_pose(self):
        """Build a geometry_msgs/Pose in WORLD_FRAME from the current params."""
        g = lambda n: float(self.get_parameter(n).value)
        p = Pose()
        p.position.x, p.position.y, p.position.z = g("x"), g("y"), g("z")
        q = quaternion_from_euler(g("roll"), g("pitch"), g("yaw"))
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
        return p

    def _param_tuple(self):
        g = lambda n: float(self.get_parameter(n).value)
        return (g("x"), g("y"), g("z"), g("roll"), g("pitch"), g("yaw"))

    def _make_camera_marker(self):
        im = InteractiveMarker()
        im.header.frame_id = WORLD_FRAME
        im.name = "camera_pose"
        im.description = "Camera pose (drag to calibrate)"
        im.scale = 0.6
        im.pose = self._param_pose()

        # always-visible handle: a red arrow along +x = the camera's viewing
        # direction in the LINK frame, so you can see where the camera looks.
        vis = InteractiveMarkerControl()
        vis.always_visible = True
        vis.interaction_mode = InteractiveMarkerControl.MOVE_ROTATE_3D
        arrow = Marker()
        arrow.type = Marker.ARROW
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.4, 0.05, 0.05
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 1.0, 0.2, 0.2, 0.9
        vis.markers.append(arrow)
        im.controls.append(vis)

        # 3 rotate rings + 3 translate arrows (canonical 6-DOF control set)
        s = math.sqrt(0.5)                                 # normalized quaternions
        for name, (ax, ay, az) in (("x", (1, 0, 0)),
                                   ("z", (0, 1, 0)),
                                   ("y", (0, 0, 1))):
            for mode, prefix in ((InteractiveMarkerControl.ROTATE_AXIS, "rotate_"),
                                 (InteractiveMarkerControl.MOVE_AXIS,   "move_")):
                c = InteractiveMarkerControl()
                c.orientation.w = s
                c.orientation.x = s * ax
                c.orientation.y = s * ay
                c.orientation.z = s * az
                c.name = prefix + name
                c.interaction_mode = mode
                im.controls.append(c)
        return im

    def _setup_interactive_marker(self):
        self._marker_dragging = False
        # last pose the marker and params agree on, so slider edits and drags
        # don't fight each other (used to push slider changes back to the marker).
        self._synced_pose = self._param_tuple()
        self.im_server = InteractiveMarkerServer(self, "overlay_calib")
        self.im_server.insert(self._make_camera_marker(),
                              feedback_callback=self._on_marker_feedback)
        self.im_server.applyChanges()

    def _apply_marker_pose(self, pose):
        """Marker pose -> x/y/z/roll/pitch/yaw params (the overlay's source of truth)."""
        p, o = pose.position, pose.orientation
        roll, pitch, yaw = euler_from_rot(quat_to_rot(o.x, o.y, o.z, o.w))
        clamp = lambda v, lo, hi: max(lo, min(hi, v))
        self.set_parameters([
            Parameter("x", Parameter.Type.DOUBLE, clamp(float(p.x), -10.0, 10.0)),
            Parameter("y", Parameter.Type.DOUBLE, clamp(float(p.y), -10.0, 10.0)),
            Parameter("z", Parameter.Type.DOUBLE, clamp(float(p.z), -10.0, 10.0)),
            Parameter("roll",  Parameter.Type.DOUBLE, float(roll)),
            Parameter("pitch", Parameter.Type.DOUBLE, float(pitch)),
            Parameter("yaw",   Parameter.Type.DOUBLE, float(yaw)),
        ])
        self._synced_pose = self._param_tuple()

    def _on_marker_feedback(self, feedback):
        et = feedback.event_type
        if et == InteractiveMarkerFeedback.MOUSE_DOWN:
            self._marker_dragging = True
        elif et == InteractiveMarkerFeedback.POSE_UPDATE:
            self._apply_marker_pose(feedback.pose)
        elif et == InteractiveMarkerFeedback.MOUSE_UP:
            self._marker_dragging = False
            self._apply_marker_pose(feedback.pose)
            x, y, z, roll, pitch, yaw = self._param_tuple()
            self.get_logger().info(
                "camera pose -> x=%.4f y=%.4f z=%.4f "
                "roll=%.4f pitch=%.4f yaw=%.4f" % (x, y, z, roll, pitch, yaw))

    def _sync_marker_from_params(self):
        """Push slider/CLI param edits back onto the marker so they stay in sync."""
        if self._marker_dragging:
            return
        cur = self._param_tuple()
        if any(abs(a - b) > 1e-5 for a, b in zip(cur, self._synced_pose)):
            self.im_server.setPose("camera_pose", self._param_pose())
            self.im_server.applyChanges()
            self._synced_pose = cur

    def _camera_matrix(self, w, h):
        qi = QTM_INTRINSIC
        fx, fy = qi["fu"] * w / qi["su"], qi["fv"] * h / qi["sv"]
        cx, cy = qi["cu"] * w / qi["su"], qi["cv"] * h / qi["sv"]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64), fx

    def _world_to_cam(self, g):
        # optical pose in world from the tunable LINK pose, then invert to world->cam
        R_opt_world = rot_from_euler(g("roll"), g("pitch"), g("yaw")) \
            @ rot_from_euler(*OPTICAL_RPY)
        Rwo = R_opt_world.T
        tcam = np.array([g("x"), g("y"), g("z")], dtype=np.float64)
        return Rwo, tcam

    def _draw_overlay(self, frame, g):
        """Project the trajectory + spheres with the full calibrated model
        (intrinsics + distortion) and draw them onto the frame."""
        h, w = frame.shape[:2]
        K, fx = self._camera_matrix(w, h)
        dist = np.array(DIST, dtype=np.float64)
        Rwo, tcam = self._world_to_cam(g)
        rvec = cv2.Rodrigues(Rwo)[0]
        tvec = (-Rwo @ tcam).reshape(3, 1)

        if self.traj_np is not None and len(self.traj_np):
            depth = (Rwo @ (self.traj_np - tcam).T).T[:, 2]
            uv = cv2.projectPoints(self.traj_np, rvec, tvec, K, dist)[0].reshape(-1, 2)
            seg = []                                   # split at points behind the camera
            for p, ok in zip(uv, depth > 0.05):
                if ok:
                    seg.append(p)
                else:
                    if len(seg) >= 2:
                        cv2.polylines(frame, [np.int32(seg)], False,
                                      TRAJ_COLOR, TRAJ_THICK, cv2.LINE_AA)
                    seg = []
            if len(seg) >= 2:
                cv2.polylines(frame, [np.int32(seg)], False,
                              TRAJ_COLOR, TRAJ_THICK, cv2.LINE_AA)

        if self.sphere_np is not None and len(self.sphere_np):
            depth = (Rwo @ (self.sphere_np - tcam).T).T[:, 2]
            uv = cv2.projectPoints(self.sphere_np, rvec, tvec, K, dist)[0].reshape(-1, 2)
            # painter's order: draw far spheres first so nearer ones occlude them
            for k in np.argsort(-depth):
                d = depth[k]
                if d <= 0.05:
                    continue
                r = fx * self.sphere_radius / d
                self._draw_sphere(frame, uv[k][0], uv[k][1], r, SPHERE_COLOR)
        return frame

    @staticmethod
    def _draw_sphere(frame, cx, cy, r, color_bgr):
        """Draw a shaded translucent "bubble": a tinted Lambert-shaded ball with
        a bright Fresnel rim and a specular highlight, alpha-composited onto the
        frame so the robot ghost still shows through the center."""
        h, w = frame.shape[:2]
        r = float(r)
        if r < 1.0:
            return
        x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
        x1, y1 = min(w, int(cx + r) + 1), min(h, int(cy + r) + 1)
        if x1 <= x0 or y1 <= y0:
            return

        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx = (xs - cx) / r
        dy = (ys - cy) / r
        rr = dx * dx + dy * dy
        inside = rr <= 1.0
        nz = np.sqrt(np.clip(1.0 - rr, 0.0, 1.0))     # sphere-surface z of the normal

        # light from upper-left-front; view straight on (+z)
        L = np.array([-0.5, -0.6, 0.62], dtype=np.float32)
        L /= np.linalg.norm(L)
        H = L + np.array([0.0, 0.0, 1.0], dtype=np.float32)
        H /= np.linalg.norm(H)
        diff = np.clip(dx * L[0] + dy * L[1] + nz * L[2], 0.0, 1.0)
        spec = np.clip(dx * H[0] + dy * H[1] + nz * H[2], 0.0, 1.0) ** 32
        fres = np.clip(1.0 - nz, 0.0, 1.0) ** 2.0     # bright at the silhouette

        base = np.array(color_bgr, dtype=np.float32)
        shade = 0.30 + 0.70 * diff                    # ambient + diffuse
        col = base[None, None, :] * shade[..., None]
        col += (255.0 * spec)[..., None]              # white specular dot
        col += (200.0 * fres)[..., None]              # glowing rim
        col = np.clip(col, 0.0, 255.0)

        alpha = 0.10 + 0.55 * fres + 0.85 * spec + 0.20 * diff
        alpha = np.clip(alpha, 0.0, SPHERE_ALPHA)
        # feather the silhouette so it doesn't alias against the scene
        edge = np.clip((1.0 - rr) * r * 0.5, 0.0, 1.0)
        alpha = np.where(inside, alpha * edge, 0.0)[..., None]

        roi = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = np.clip(col * alpha + roi * (1.0 - alpha),
                                      0, 255).astype(np.uint8)

    def tick(self):
        now = self.get_clock().now().to_msg()

        # keep the interactive marker aligned with any slider/CLI param edits
        self._sync_marker_from_params()

        # --- tunable camera TF: WORLD -> LINK ---
        g = lambda n: float(self.get_parameter(n).value)
        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = WORLD_FRAME
        tf.child_frame_id  = LINK_FRAME
        tf.transform.translation.x = g("x")
        tf.transform.translation.y = g("y")
        tf.transform.translation.z = g("z")
        q = quaternion_from_euler(g("roll"), g("pitch"), g("yaw"))
        tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = q
        self.tf_bc.sendTransform(tf)

        # --- trajectory marker in WORLD ---
        m = Marker()
        m.header.frame_id = WORLD_FRAME
        m.header.stamp = now
        m.ns, m.id = "trajectory", 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.03
        m.color.g, m.color.a = 1.0, 1.0
        m.pose.orientation.w = 1.0
        m.points = [Point(x=float(px), y=float(py), z=float(pz)) for px, py, pz in self.trajectory]
        self.mark_pub.publish(m)

        # --- safety_filter robot sphere, re-stamped to the image time ---
        if self.robot_marker is not None:
            self.robot_marker.header.stamp = now
            self.robot_pub.publish(self.robot_marker)

        # --- static still background (fresh copy each tick to redraw overlay) ---
        if self.still is None:
            return
        frame = self.still.copy()
        h, w = frame.shape[:2]

        if DRAW_OVERLAY:
            frame = self._draw_overlay(frame, g)

        img = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img.header.stamp = now
        img.header.frame_id = OPTICAL_FRAME
        self.img_pub.publish(img)

        # --- camera_info, same stamp + frame ---
        info = CameraInfo()
        info.header.stamp = now
        info.header.frame_id = OPTICAL_FRAME
        info.width, info.height = w, h
        info.distortion_model = "plumb_bob"
        info.d = [float(v) for v in DIST]
        # QTM sensor units -> pixels, scaled to the actual frame size.
        qi = QTM_INTRINSIC
        fx, fy = qi["fu"] * w / qi["su"], qi["fv"] * h / qi["sv"]
        cx, cy = qi["cu"] * w / qi["su"], qi["cv"] * h / qi["sv"]
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(info)


def main():
    rclpy.init()
    node = OverlayCalib()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


#  6:26:50 - 4:43 = 6:22:07 + ::72 = 6:23:19