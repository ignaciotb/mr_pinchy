from __future__ import annotations

from functools import lru_cache

import numpy as np
from geometry_msgs.msg import WrenchStamped
from mavros_msgs.msg import OverrideRCIn


_BLUEROV2_HEAVY_ROTORS_60002 = [
    #  AX   AY   AZ    PX    PY    PZ   KM
    (1.0, -1.0, 0.0, 0.14, 0.10, 0.06, 0.0),  # rotor 0
    (1.0, 1.0, 0.0, 0.14, -0.10, 0.06, 0.0),  # rotor 1
    (1.0, 1.0, 0.0, -0.14, 0.10, 0.06, 0.0),  # rotor 2
    (1.0, -1.0, 0.0, -0.14, -0.10, 0.06, 0.0),  # rotor 3
    (0.0, 0.0, -1.0, 0.12, 0.22, 0.00, 0.0),  # rotor 4
    (0.0, 0.0, 1.0, 0.12, -0.22, 0.00, 0.0),  # rotor 5
    (0.0, 0.0, 1.0, -0.12, 0.22, 0.00, 0.0),  # rotor 6
    (0.0, 0.0, -1.0, -0.12, -0.22, 0.00, 0.0),  # rotor 7
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


def build_bluerov2_heavy_effectiveness_matrix(
    ct: float | np.ndarray | None = None,
) -> np.ndarray:
    """Build the 6×8 effectiveness matrix B from BlueROV2 Heavy geometry.

    Mirrors ActuatorEffectivenessRotors::computeEffectivenessMatrix().

    Parameters
    ----------
    ct : float | np.ndarray | None
        Rotor thrust coefficient(s) used as in PX4:
          thrust = ct * axis
          moment = ct * cross(position, axis) - ct * km * axis
        If None, uses PX4 default CA_ROTOR{i}_CT = 6.5 for all 8 rotors.
        If float, applies same value to every rotor.
        If array-like, must provide one value per rotor.
    Returns
    -------
    B : np.ndarray, shape (6, 8)
        Rows 0-2 = torque effectiveness  [roll, pitch, yaw]
        Rows 3-5 = force  effectiveness  [Fx,   Fy,    Fz]
    """
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

    B = np.zeros((6, n_rotors))

    for i, (ax, ay, az, px, py, pz, km) in enumerate(rotors):
        ct_i = ct_per_rotor[i]
        axis = np.array([ax, ay, az])
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            continue
        axis = axis / norm

        position = np.array([px, py, pz])

        thrust = ct_i * axis
        moment = ct_i * np.cross(position, axis) - ct_i * km * axis

        B[0:3, i] = moment
        B[3:6, i] = thrust

    return B


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
    """Translate a body-FRD wrench into ArduSub RC override channels.

    Input wrench ordering follows PX4 control-allocation convention:
        [tau_x, tau_y, tau_z, F_x, F_y, F_z]

    The returned RC axes follow default ArduSub channel mapping:
        ch0 pitch, ch1 roll, ch2 throttle, ch3 yaw, ch4 forward, ch5 lateral
    """

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
    channels[0] = _pwm_axis(rc_axes[1], neutral_pwm, delta_pwm)  # pitch
    channels[1] = _pwm_axis(rc_axes[0], neutral_pwm, delta_pwm)  # roll
    channels[2] = _pwm_axis(rc_axes[3], neutral_pwm, delta_pwm)  # throttle
    channels[3] = _pwm_axis(rc_axes[2], neutral_pwm, delta_pwm)  # yaw
    channels[4] = _pwm_axis(rc_axes[4], neutral_pwm, delta_pwm)  # forward
    channels[5] = _pwm_axis(rc_axes[5], neutral_pwm, delta_pwm)  # lateral

    rc_msg = OverrideRCIn()
    rc_msg.channels = channels
    return rc_msg


if __name__ == "__main__":
    # Example usage:
    wrench_msg = WrenchStamped()
    wrench_msg.wrench.torque.z = 10.0
    rc_override = wrench_to_rc_override(wrench_msg)
    print(rc_override.channels)
