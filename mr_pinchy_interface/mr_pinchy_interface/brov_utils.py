from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped, Vector3Stamped, WrenchStamped
from mavros_msgs.msg import OverrideRCIn
from nav_msgs.msg import Odometry


_ENU_TO_NED_ROTATION = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)

_BLUEROV2_HEAVY_ROTORS_60002 = [
    #  AX   AY   AZ    PX    PY    PZ   KM
    (1.0, -1.0, 0.0, 0.14, 0.10, 0.06, 0.0),
    (1.0, 1.0, 0.0, 0.14, -0.10, 0.06, 0.0),
    (1.0, 1.0, 0.0, -0.14, 0.10, 0.06, 0.0),
    (1.0, -1.0, 0.0, -0.14, -0.10, 0.06, 0.0),
    (0.0, 0.0, -1.0, 0.12, 0.22, 0.00, 0.0),
    (0.0, 0.0, 1.0, 0.12, -0.22, 0.00, 0.0),
    (0.0, 0.0, 1.0, -0.12, 0.22, 0.00, 0.0),
    (0.0, 0.0, -1.0, -0.12, -0.22, 0.00, 0.0),
]

_DEFAULT_MOTOR_DIRECTION_SIGNS = (-1, -1, 1, 1, -1, 1, 1, -1)

# Raw ArduSub VECTORED_6DOF mixer in axis order:
# [roll, pitch, yaw, throttle, forward, lateral]
_ARDUSUB_HEAVY_AXIS_TO_MOTOR = np.array(
    [
        [0.0, 0.0, 1.0, 0.0, -1.0, 1.0],
        [0.0, 0.0, -1.0, 0.0, -1.0, -1.0],
        [0.0, 0.0, -1.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, -1.0],
        [1.0, -1.0, 0.0, -1.0, 0.0, 0.0],
        [-1.0, -1.0, 0.0, -1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, -1.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, -1.0, 0.0, 0.0],
    ],
    dtype=float,
)

_DEFAULT_CT = 6.5


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quat_to_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return x, y, z, w


def quat_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def rpy_from_rotation_matrix(rotation: np.ndarray) -> tuple[float, float, float]:
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    pitch = math.asin(clamp(-rotation[2, 0], -1.0, 1.0))
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return roll, pitch, yaw


def world_to_ned_rotation(frame_id: str) -> np.ndarray:
    normalized = frame_id.rsplit("/", 1)[-1].lower()
    if normalized == "ned" or normalized.endswith("_ned"):
        return np.eye(3, dtype=float)
    return _ENU_TO_NED_ROTATION


def pose_rpy_in_ned(pose, *, world_frame: str) -> tuple[float, float, float]:
    rotation_world_body = quat_to_rotation_matrix(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    rotation_ned_body = world_to_ned_rotation(world_frame) @ rotation_world_body
    return rpy_from_rotation_matrix(rotation_ned_body)


def vector3_to_array(msg) -> np.ndarray:
    return np.array([msg.x, msg.y, msg.z], dtype=float)


def vector3_stamped(msg, *, frame_id: str, stamp=None) -> Vector3Stamped:
    vector = Vector3Stamped()
    vector.header.frame_id = frame_id
    if stamp is not None:
        vector.header.stamp = stamp
    vector.vector.x = msg.x
    vector.vector.y = msg.y
    vector.vector.z = msg.z
    return vector


def pose_stamped_from_pose(pose, *, frame_id: str, stamp=None) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.pose = pose
    return msg


def pose_stamped_from_transform(transform: TransformStamped) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = transform.header.frame_id
    msg.header.stamp = transform.header.stamp
    msg.pose.position.x = transform.transform.translation.x
    msg.pose.position.y = transform.transform.translation.y
    msg.pose.position.z = transform.transform.translation.z
    msg.pose.orientation.x = transform.transform.rotation.x
    msg.pose.orientation.y = transform.transform.rotation.y
    msg.pose.orientation.z = transform.transform.rotation.z
    msg.pose.orientation.w = transform.transform.rotation.w
    return msg


def transform_from_pose(
    pose,
    *,
    target_frame: str,
    source_frame: str,
    stamp=None,
) -> TransformStamped:
    transform = TransformStamped()
    transform.header.frame_id = target_frame
    transform.child_frame_id = source_frame
    if stamp is not None:
        transform.header.stamp = stamp
    transform.transform.translation.x = pose.position.x
    transform.transform.translation.y = pose.position.y
    transform.transform.translation.z = pose.position.z
    transform.transform.rotation.x = pose.orientation.x
    transform.transform.rotation.y = pose.orientation.y
    transform.transform.rotation.z = pose.orientation.z
    transform.transform.rotation.w = pose.orientation.w
    return transform


def twist_to_body_velocity(msg: TwistStamped) -> np.ndarray:
    return np.array(
        [
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
            msg.twist.angular.x,
            msg.twist.angular.y,
            msg.twist.angular.z,
        ],
        dtype=float,
    )


def odom_to_body_state(odom: Odometry) -> tuple[np.ndarray, np.ndarray]:
    pose = odom.pose.pose
    twist = odom.twist.twist
    roll, pitch, yaw = quat_to_rpy(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    eta = np.array(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            roll,
            pitch,
            yaw,
        ],
        dtype=float,
    )
    nu = np.array(
        [
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z,
        ],
        dtype=float,
    )
    return eta, nu


@dataclass
class PIDAxis:
    kp: float
    ki: float
    kd: float
    integral_limit: float
    output_limit: float
    integral: float = 0.0
    prev_error: float = 0.0
    has_prev: bool = False

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.has_prev = False

    def step(self, error: float, dt: float, *, wrap_error: bool = False) -> float:
        if wrap_error:
            error = wrap_pi(error)

        if dt <= 1e-6:
            deriv = 0.0
        else:
            self.integral += error * dt
            self.integral = clamp(self.integral, -self.integral_limit, self.integral_limit)
            if self.has_prev:
                error_delta = error - self.prev_error
                if wrap_error:
                    error_delta = wrap_pi(error_delta)
                deriv = error_delta / dt
            else:
                deriv = 0.0

        out = self.kp * error + self.ki * self.integral + self.kd * deriv
        out = clamp(out, -self.output_limit, self.output_limit)
        self.prev_error = error
        self.has_prev = True
        return out


def neutral_rc_override(*, neutral_pwm: int) -> OverrideRCIn:
    msg = OverrideRCIn()
    channels = [OverrideRCIn.CHAN_NOCHANGE] * 18
    for idx in range(6):
        channels[idx] = neutral_pwm
    msg.channels = channels
    return msg


def build_wrench_message(
    *,
    force_x: float = 0.0,
    force_y: float = 0.0,
    force_z: float = 0.0,
    torque_x: float = 0.0,
    torque_y: float = 0.0,
    torque_z: float = 0.0,
    frame_id: str = "",
    stamp=None,
) -> WrenchStamped:
    msg = WrenchStamped()
    if hasattr(msg, "header"):
        msg.header.frame_id = frame_id
        if stamp is not None:
            msg.header.stamp = stamp
    msg.wrench.force.x = force_x
    msg.wrench.force.y = force_y
    msg.wrench.force.z = force_z
    msg.wrench.torque.x = torque_x
    msg.wrench.torque.y = torque_y
    msg.wrench.torque.z = torque_z
    return msg


def pid_outputs_to_wrench(
    *,
    surge_pid_output: float,
    sway_pid_output: float,
    heave_pid_output: float,
    roll_pid_output: float = 0.0,
    pitch_pid_output: float = 0.0,
    yaw_pid_output: float,
    force_x_scale: float,
    force_y_scale: float,
    force_z_scale: float,
    torque_x_scale: float = 1.0,
    torque_y_scale: float = 1.0,
    torque_z_scale: float,
    frame_id: str = "",
    stamp=None,
) -> WrenchStamped:
    return build_wrench_message(
        force_x=surge_pid_output * force_x_scale,
        force_y=sway_pid_output * force_y_scale,
        force_z=heave_pid_output * force_z_scale,
        torque_x=roll_pid_output * torque_x_scale,
        torque_y=pitch_pid_output * torque_y_scale,
        torque_z=yaw_pid_output * torque_z_scale,
        frame_id=frame_id,
        stamp=stamp,
    )


def body_wrench_message(
    tau: np.ndarray,
    *,
    frame_id: str = "mr_pinchy/base_link",
    stamp=None,
) -> WrenchStamped:
    tau = np.asarray(tau, dtype=float)
    return build_wrench_message(
        force_x=float(tau[0]),
        force_y=float(tau[1]),
        force_z=float(tau[2]),
        torque_x=float(tau[3]),
        torque_y=float(tau[4]),
        torque_z=float(tau[5]),
        frame_id=frame_id,
        stamp=stamp,
    )


def build_bluerov2_heavy_effectiveness_matrix(
    ct: float | np.ndarray | None = None,
) -> np.ndarray:
    rotors = _BLUEROV2_HEAVY_ROTORS_60002
    n_rotors = len(rotors)
    if ct is None:
        ct_per_rotor = np.full(n_rotors, _DEFAULT_CT)
    elif np.isscalar(ct):
        ct_per_rotor = np.full(n_rotors, float(ct))
    else:
        ct_per_rotor = np.asarray(ct, dtype=float)
        if ct_per_rotor.shape != (n_rotors,):
            raise ValueError(
                f"ct must have shape ({n_rotors},), got {ct_per_rotor.shape}"
            )

    effectiveness = np.zeros((6, n_rotors))
    for index, (ax, ay, az, px, py, pz, km) in enumerate(rotors):
        ct_i = ct_per_rotor[index]
        axis = np.array([ax, ay, az], dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            continue
        axis = axis / norm

        position = np.array([px, py, pz], dtype=float)
        thrust = ct_i * axis
        moment = ct_i * np.cross(position, axis) - ct_i * km * axis

        effectiveness[0:3, index] = moment
        effectiveness[3:6, index] = thrust

    return effectiveness


def _pwm_axis(axis_cmd: float, neutral_pwm: int, delta_pwm: int) -> int:
    clipped = float(np.clip(axis_cmd, -1.0, 1.0))
    return int(round(neutral_pwm + clipped * delta_pwm))


@lru_cache(maxsize=None)
def _build_wrench_to_rc_matrix(
    motor_direction_signs: tuple[int, ...],
) -> np.ndarray:
    if len(motor_direction_signs) != 8:
        raise ValueError(
            "motor_direction_signs must contain 8 entries for the BlueROV2 Heavy thrusters"
        )

    direction = np.diag(np.asarray(motor_direction_signs, dtype=float))
    effectiveness = build_bluerov2_heavy_effectiveness_matrix()
    composite = effectiveness @ direction @ _ARDUSUB_HEAVY_AXIS_TO_MOTOR
    return np.linalg.pinv(composite)


def wrench_to_rc_override(
    msg: WrenchStamped,
    neutral_pwm: int = 1500,
    delta_pwm: int = 100,
    motor_direction_signs: tuple[int, ...] = _DEFAULT_MOTOR_DIRECTION_SIGNS,
) -> OverrideRCIn:
    if delta_pwm <= 0:
        raise ValueError("delta_pwm must be positive")

    wrench = np.array(
        [
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z,
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
        ],
        dtype=float,
    )
    rc_axes = _build_wrench_to_rc_matrix(tuple(motor_direction_signs)) @ wrench
    rc_axes = np.clip(rc_axes, -1.0, 1.0)

    channels = [OverrideRCIn.CHAN_NOCHANGE] * 18
    channels[0] = _pwm_axis(rc_axes[1], neutral_pwm, delta_pwm)
    channels[1] = _pwm_axis(rc_axes[0], neutral_pwm, delta_pwm)
    channels[2] = _pwm_axis(rc_axes[3], neutral_pwm, delta_pwm)
    channels[3] = _pwm_axis(rc_axes[2], neutral_pwm, delta_pwm)
    channels[4] = _pwm_axis(rc_axes[4], neutral_pwm, delta_pwm)
    channels[5] = _pwm_axis(rc_axes[5], neutral_pwm, delta_pwm)

    rc_msg = OverrideRCIn()
    rc_msg.channels = channels
    return rc_msg
