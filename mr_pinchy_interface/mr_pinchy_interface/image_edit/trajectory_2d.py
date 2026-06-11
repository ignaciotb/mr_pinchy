#!/usr/bin/env python3
"""
trajectory_2d.py

Read the same bag as overlay_rviz.py and draw a *2D top-down* (x-y) trajectory
of mr_pinchy/base_link_ekf in the `map` frame with matplotlib (no RViz).

On top of the path we stamp commanded-velocity arrows: at a few bag-relative
times we pick the nearest /mr_pinchy/cmd_vel_ref (a TwistStamped in the robot's
base_link_ekf_flu body frame), rotate its linear velocity into `map`, and draw
it as an arrow rooted at the robot's position at that moment.

The path itself is colored as a *heatmap* of how much the CBF safety filter
modified the command: at every trajectory point we take the linear-velocity
difference ‖cmd_vel − cmd_vel_ref‖ (filtered minus reference), so hot/opaque
stretches are where the safety filter intervened hardest.

Frames:
    odom (/mocap/mr_pinchy/odom_ekf) lives in `mocap`; we reproject the path
    into `map` via the bag's TF tree (mocap -> map). cmd_vel_ref lives in
    `mr_pinchy/base_link_ekf_flu`; we rotate it (rotation only, it's a velocity)
    into `map` via the TF at that timestamp.

Run (needs a sourced ROS2 humble env with rosbag2_py + tf2):
    python3 trajectory_2d.py
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")          # write a PNG without needing a display
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
plt.style.use('science')

# ============================ CONFIG — edit these ============================
ROSBAG_PATH   = "/home/shaohang/workspace/underwater_ws/data_rosbag/cut.mcap"
ODOM_TOPIC    = "/mocap/mr_pinchy/odom_ekf"
CMD_REF_TOPIC = "/mr_pinchy/cmd_vel_ref"   # reference command (arrows come from this)
CMD_FILT_TOPIC = "/mr_pinchy/cmd_vel"      # CBF-filtered command actually sent
MAP_FRAME     = "map"

TRAJ_STRIDE = 5            # keep every Nth odom sample (downsample the line)

# Bag-relative times (s after the first odom sample) to stamp a cmd_vel arrow.
# These match video_overlay.py FRAME_TIMES so the arrows line up with the ghosts.
ARROW_TIMES = [4.0, 12.0, 20.0, 31.0]   # 31s ~ last cmd_vel_ref sample
TRAJ_START  = 3.0                        # bag-relative start time of the plotted path

ARROW_SCALE = 2.0         # metres of arrow drawn per (m/s) of commanded speed
OUT_IMAGE   = "/home/shaohang/workspace/underwater_ws/src/mr_pinchy/mr_pinchy_interface/mr_pinchy_interface/image_edit/trajectory_2d.svg"
# ===========================================================================


def quat_rotate(q, v):
    """Rotate vector v=(x,y,z) by quaternion q=(x,y,z,w)."""
    x, y, z, w = q
    vx, vy, vz = v
    # t = 2 * cross(q_xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # v' = v + w*t + cross(q_xyz, t)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return rx, ry, rz


def _nearest(times, t):
    """Index of the entry in sorted array `times` closest to scalar `t`."""
    j = int(np.searchsorted(times, t))
    j = min(max(j, 1), len(times) - 1)
    return j - 1 if (t - times[j - 1]) <= (times[j] - t) else j


def read_bag():
    """Return (traj_xy [Nx2 map], arrows [(t,(x,y),(vx,vy),speed)...], mod [N]).

    `mod[i]` is ‖cmd_vel − cmd_vel_ref‖ (linear, m/s) at trajectory point i —
    how much the CBF safety filter changed the command there.
    """
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from tf2_msgs.msg import TFMessage
    from geometry_msgs.msg import PointStamped
    from tf2_geometry_msgs import do_transform_point
    import tf2_ros
    import rclpy.time
    from rclpy.duration import Duration

    reader = SequentialReader()
    reader.open(StorageOptions(uri=ROSBAG_PATH, storage_id="mcap"),
                ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    odom_msg = get_message(types[ODOM_TOPIC])
    cmd_msg  = get_message(types[CMD_REF_TOPIC])

    # big cache so no transform from the 33 s bag gets evicted before lookups
    buf = tf2_ros.Buffer(cache_time=Duration(seconds=120))
    odoms, refs, filts, i = [], [], [], 0
    while reader.has_next():
        tname, data, _ = reader.read_next()
        if tname in ("/tf", "/tf_static"):
            for tr in deserialize_message(data, TFMessage).transforms:
                if tname == "/tf_static":
                    buf.set_transform_static(tr, "bag")
                else:
                    buf.set_transform(tr, "bag")
        elif tname == ODOM_TOPIC:
            if i % TRAJ_STRIDE == 0:
                m = deserialize_message(data, odom_msg)
                t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                odoms.append((t, m.header, m.pose.pose.position))
            i += 1
        elif tname == CMD_REF_TOPIC:
            m = deserialize_message(data, cmd_msg)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            refs.append((t, m.header.frame_id, m.twist.linear))
        elif tname == CMD_FILT_TOPIC:
            m = deserialize_message(data, cmd_msg)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            filts.append((t, m.twist.linear))

    if not odoms:
        raise RuntimeError("no odom samples in bag")
    t0 = odoms[0][0]

    # cmd_vel_ref stops ~2 s before the bag ends; truncate the trajectory to the
    # last reference timestamp so the path doesn't run past where we have a
    # commanded reference to compare against.
    t_end = max(r[0] for r in refs)
    odoms = [o for o in odoms if t0 + TRAJ_START <= o[0] <= t_end]

    def to_map_point(header, p):
        ps = PointStamped()
        ps.header = header
        ps.point.x, ps.point.y, ps.point.z = p.x, p.y, p.z
        try:
            tf = buf.lookup_transform(MAP_FRAME, header.frame_id,
                                      rclpy.time.Time.from_msg(header.stamp))
        except Exception:
            tf = buf.lookup_transform(MAP_FRAME, header.frame_id, rclpy.time.Time())
        q = do_transform_point(ps, tf).point
        return q.x, q.y

    traj = np.array([to_map_point(h, p) for _, h, p in odoms])

    # --- per-point CBF modification magnitude: ‖filtered - reference‖ (linear) ---
    ref_t = np.array([r[0] for r in refs])
    ref_v = np.array([[r[2].x, r[2].y, r[2].z] for r in refs])
    flt_t = np.array([f[0] for f in filts])
    flt_v = np.array([[f[1].x, f[1].y, f[1].z] for f in filts])
    mod = np.array([
        np.linalg.norm(flt_v[_nearest(flt_t, ot)] - ref_v[_nearest(ref_t, ot)])
        for ot, _, _ in odoms])

    # one cmd_vel_ref arrow per requested bag-relative time. Skip the trailing
    # zero "stop" command (cmd_vel_ref ends ~31s with a 0 vector) so the t=33
    # arrow shows the last real heading instead of vanishing.
    ref_spd = np.linalg.norm(ref_v, axis=1)
    moving = np.where(ref_spd > 0.02)[0]
    arrows = []
    for rel in ARROW_TIMES:
        target = t0 + rel
        oj = min(range(len(odoms)), key=lambda j: abs(odoms[j][0] - target))
        px, py = traj[oj]
        cj = int(moving[_nearest(ref_t[moving], target)])
        cstamp, cframe, lin = refs[cj]
        # rotate the body-frame velocity into map (rotation only)
        try:
            tf = buf.lookup_transform(
                MAP_FRAME, cframe,
                rclpy.time.Time(nanoseconds=int(cstamp * 1e9)))
        except Exception:
            tf = buf.lookup_transform(MAP_FRAME, cframe, rclpy.time.Time())
        r = tf.transform.rotation
        vx, vy, vz = quat_rotate((r.x, r.y, r.z, r.w),
                                 (lin.x, lin.y, lin.z))
        speed = math.sqrt(lin.x**2 + lin.y**2 + lin.z**2)
        arrows.append((rel, (px, py), (vx, vy), speed))
    return traj, arrows, mod


TRAJ_THICK  = 6.0
ARROW_COLOR = (1.0, 0.333, 0.333)                 # dark, same for all arrows
# yellow-ish ramp with strong contrast: deep amber = no modification ->
# near-white pale yellow = max modification
HEAT_LOW    = (0.0, 0.0, 0.0)                      # normal trajectory (teal-green)
HEAT_HIGH   = (1.0, 1.0, 0.0)                      # max modification (yellow #ffff00)
HEAT_GAMMA  = 0.6                                       # <1 spreads the clustered low values


def plot(traj, arrows, mod):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import PowerNorm, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable

    cmap = LinearSegmentedColormap.from_list("cbf_mod", [HEAT_LOW, HEAT_HIGH])

    # size the figure to the trajectory's aspect ratio so it isn't letterboxed
    xspan = traj[:, 0].max() - traj[:, 0].min()
    yspan = traj[:, 1].max() - traj[:, 1].min()
    base, pad = 8.0, 1.15
    if yspan >= xspan:
        figsize = (base * max(xspan / yspan, 0.35) * pad, base)
    else:
        figsize = (base, base * max(yspan / xspan, 0.35) * pad)

    fig, ax = plt.subplots(figsize=figsize)

    # trajectory colored by CBF modification: dark where unmodified, light where
    # the safety filter changed the command the most.
    segs = np.stack([traj[:-1], traj[1:]], axis=1)
    segval = 0.5 * (mod[:-1] + mod[1:])
    vmax = float(mod.max()) or 1.0
    norm = PowerNorm(HEAT_GAMMA, vmin=0.0, vmax=vmax)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=TRAJ_THICK,
                        capstyle="round", zorder=2)
    lc.set_array(segval)
    ax.add_collection(lc)

    for rel, (px, py), (vx, vy), speed in arrows:
        if speed > 1e-3:        # no dots; just the reference-velocity arrows
            ax.arrow(px, py, vx * ARROW_SCALE, vy * ARROW_SCALE,
                     head_width=0.08, head_length=0.18,
                     fc=ARROW_COLOR, ec=ARROW_COLOR,
                     length_includes_head=True, lw=2.0, zorder=6)

    # cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
    #                     ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label("CBF modification  ‖cmd_vel − cmd_vel_ref‖  [m/s]")

    ax.set_xlabel("$p_x$ (m)", fontsize=28)
    ax.set_ylabel("$p_y$ (m)", fontsize=28)
    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()
    # rotate the plot 180 deg about its center (flip both axes), labels stay upright
    ax.invert_xaxis()
    ax.invert_yaxis()
    # ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_IMAGE, dpi=150)
    print(f"Wrote {OUT_IMAGE}")
    print(f"Arrow scale: {ARROW_SCALE} m drawn per 1 m/s commanded")


def main():
    traj, arrows, mod = read_bag()
    print(f"Loaded {len(traj)} trajectory points in '{MAP_FRAME}' frame")
    print(f"CBF modification ‖cmd_vel-cmd_vel_ref‖: "
          f"max={mod.max():.3f}  mean={mod.mean():.3f} m/s")
    for rel, (px, py), (vx, vy), speed in arrows:
        print(f"  t={rel:5.1f}s  pos=({px:.2f},{py:.2f})  "
              f"v_map=({vx:.3f},{vy:.3f})  |v|={speed:.3f} m/s")
    plot(traj, arrows, mod)


if __name__ == "__main__":
    main()
