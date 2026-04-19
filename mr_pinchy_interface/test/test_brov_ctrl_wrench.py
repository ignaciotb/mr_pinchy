from __future__ import annotations

import sys
import types
from pathlib import Path


class _Vector3:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class _Wrench:
    def __init__(self) -> None:
        self.force = _Vector3()
        self.torque = _Vector3()


class WrenchStamped:
    def __init__(self) -> None:
        self.wrench = _Wrench()


class OverrideRCIn:
    CHAN_NOCHANGE = 65535

    def __init__(self) -> None:
        self.channels = []


geometry_msgs = types.ModuleType("geometry_msgs")
geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
geometry_msgs_msg.WrenchStamped = WrenchStamped
geometry_msgs.msg = geometry_msgs_msg
sys.modules["geometry_msgs"] = geometry_msgs
sys.modules["geometry_msgs.msg"] = geometry_msgs_msg

mavros_msgs = types.ModuleType("mavros_msgs")
mavros_msgs_msg = types.ModuleType("mavros_msgs.msg")
mavros_msgs_msg.OverrideRCIn = OverrideRCIn
mavros_msgs.msg = mavros_msgs_msg
sys.modules["mavros_msgs"] = mavros_msgs
sys.modules["mavros_msgs.msg"] = mavros_msgs_msg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mr_pinchy.mr_pinchy_interface.mr_pinchy_interface.brov_wrench_to_pwm import (
    build_bluerov2_heavy_effectiveness_matrix,
    wrench_to_rc_override,
)


def _make_wrench(
    *,
    tau_x: float = 0.0,
    tau_y: float = 0.0,
    tau_z: float = 0.0,
    force_x: float = 0.0,
    force_y: float = 0.0,
    force_z: float = 0.0,
) -> WrenchStamped:
    msg = WrenchStamped()
    msg.wrench.torque.x = tau_x
    msg.wrench.torque.y = tau_y
    msg.wrench.torque.z = tau_z
    msg.wrench.force.x = force_x
    msg.wrench.force.y = force_y
    msg.wrench.force.z = force_z
    return msg


def test_zero_wrench_gives_neutral_channels() -> None:
    msg = wrench_to_rc_override(_make_wrench())

    assert msg.channels[:6] == [1500] * 6
    assert msg.channels[6:] == [OverrideRCIn.CHAN_NOCHANGE] * 12


def test_pure_roll_wrench_maps_to_roll_channel() -> None:
    msg = wrench_to_rc_override(_make_wrench(tau_x=1.0))

    assert msg.channels[1] == 1517
    assert msg.channels[0] == 1500
    assert msg.channels[2] == 1500
    assert msg.channels[3] == 1500
    assert msg.channels[4] == 1500
    assert msg.channels[5] == 1500


def test_pure_pitch_wrench_maps_to_pitch_channel() -> None:
    msg = wrench_to_rc_override(_make_wrench(tau_y=1.0))

    assert msg.channels[0] == 1532
    assert msg.channels[1] == 1500
    assert msg.channels[2] == 1500
    assert msg.channels[3] == 1500
    assert msg.channels[4] == 1500
    assert msg.channels[5] == 1500


def test_pure_yaw_wrench_maps_to_yaw_channel() -> None:
    msg = wrench_to_rc_override(_make_wrench(tau_z=1.0))

    assert msg.channels[3] == 1523
    assert msg.channels[0] == 1500
    assert msg.channels[1] == 1500
    assert msg.channels[2] == 1500
    assert msg.channels[4] == 1500
    assert msg.channels[5] == 1500


def test_positive_frd_heave_lowers_throttle_pwm() -> None:
    msg = wrench_to_rc_override(_make_wrench(force_z=1.0))

    assert msg.channels[2] == 1496
    assert msg.channels[0] == 1500
    assert msg.channels[1] == 1500
    assert msg.channels[3] == 1500
    assert msg.channels[4] == 1500
    assert msg.channels[5] == 1500


def test_pure_surge_is_forward_dominant_with_small_pitch_compensation() -> None:
    msg = wrench_to_rc_override(_make_wrench(force_x=1.0))

    assert msg.channels[4] == 1505
    assert msg.channels[0] == 1498
    assert msg.channels[1] == 1500
    assert msg.channels[2] == 1500
    assert msg.channels[3] == 1500
    assert msg.channels[5] == 1500


def test_pure_sway_is_lateral_dominant_with_small_roll_compensation() -> None:
    msg = wrench_to_rc_override(_make_wrench(force_y=1.0))

    assert msg.channels[5] == 1505
    assert msg.channels[1] == 1501
    assert msg.channels[0] == 1500
    assert msg.channels[2] == 1500
    assert msg.channels[3] == 1500
    assert msg.channels[4] == 1500


def test_large_input_clips_to_pwm_span() -> None:
    msg = wrench_to_rc_override(_make_wrench(tau_y=10.0))

    assert msg.channels[0] == 1600


def test_motor_direction_signs_change_output_deterministically() -> None:
    default = wrench_to_rc_override(_make_wrench(tau_z=1.0))
    flipped = wrench_to_rc_override(
        _make_wrench(tau_z=1.0),
        motor_direction_signs=(1, 1, 1, 1, 1, 1, 1, 1),
    )

    assert default.channels != flipped.channels


def test_effectiveness_matrix_supports_1022_geometry() -> None:
    B_60002 = build_bluerov2_heavy_effectiveness_matrix(geometry="60002")
    B_1022 = build_bluerov2_heavy_effectiveness_matrix(geometry="1022")

    assert B_60002.shape == (6, 8)
    assert B_1022.shape == (6, 8)
    assert not (B_60002 == B_1022).all()
