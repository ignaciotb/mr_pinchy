# Wrench to RC Override Design

This note documents the implemented design in `brov_ctrl_wrench.py` for translating
`geometry_msgs/WrenchStamped` into `mavros_msgs/OverrideRCIn`.

The translator is a pure function:

```python
wrench_to_rc_override(
    msg: WrenchStamped,
    *,
    neutral_pwm: int = 1500,
    delta_pwm: int = 100,
    geometry: str = "60002",
    motor_direction_signs: tuple[int, ...] = (-1, -1, 1, 1, -1, 1, 1, -1),
) -> OverrideRCIn
```

It does not create a ROS node, does not use TF, and does not perform any frame conversion.

## Input Contract

The input wrench is assumed to already be expressed in the vehicle body `FRD` frame:

- `+X`: forward
- `+Y`: right
- `+Z`: down
- positive torque follows the right-hand rule about those body axes

The control vector is read as:

```text
w = [tau_x, tau_y, tau_z, F_x, F_y, F_z]
    [ roll,  pitch, yaw,  surge, sway, heave ]
```

This matches the PX4-side convention described in:

- `wrench_controller_design.md`
- `uuv_wrench_allocator.py`

No attempt is made to support `FLU`, map/world frames, or `WrenchStamped.header.frame_id`.

### Axis meaning used by the translator

For the input wrench:

- `+tau_x`: positive roll torque
- `+tau_y`: positive pitch torque
- `+tau_z`: positive yaw torque
- `+F_x`: forward surge force
- `+F_y`: rightward sway force
- `+F_z`: downward heave force

For the ArduSub RC axes produced by the translator:

- positive `roll` means roll right
- positive `pitch` means pitch up
- positive `yaw` means yaw right, clockwise when viewed from above
- positive `forward` means surge forward
- positive `lateral` means sway right
- positive `throttle` means heave up

The important mismatch is heave:

- input wrench uses `FRD`, so positive `F_z` means down
- ArduSub throttle uses pilot semantics, so positive throttle means up

That is why the implemented `F_z -> throttle` mapping has the opposite sign.

## Output Contract

The output is a default ArduSub-style `OverrideRCIn` message:

- `channels[0]`: pitch
- `channels[1]`: roll
- `channels[2]`: throttle
- `channels[3]`: yaw
- `channels[4]`: forward
- `channels[5]`: lateral
- `channels[6..17]`: `CHAN_NOCHANGE`

This assumes the default ArduSub `RCMAP_*` layout documented in `bodycmd_mixer.md`.

The design is intended for BlueROV2 Heavy in `MANUAL`-style direct stick mixing. In
assisted modes, RC override is no longer a direct body-axis command interface.

## Design Idea

The translator composes two existing models instead of inventing a new hand-tuned mapping:

1. PX4 wrench-to-thruster geometry model
2. ArduSub RC-axis-to-thruster mixer model

### 1. PX4 effectiveness matrix

From `uuv_wrench_allocator.py`, build the BlueROV2 Heavy effectiveness matrix:

```text
B : 6 x 8
```

such that:

```text
w = B u_motor
```

where `u_motor` is the normalized 8-thruster command vector.

### 2. ArduSub Heavy RC mixer

From `bodycmd_mixer.md`, the raw BlueROV2 Heavy `VECTORED_6DOF` mixer can be written as:

```text
u_motor_raw = A a_rc
```

where:

```text
a_rc = [roll, pitch, yaw, throttle, forward, lateral]
```

and:

```text
      [ 0  0  1  0 -1  1 ]
      [ 0  0 -1  0 -1 -1 ]
  A = [ 0  0 -1  0  1  1 ]
      [ 0  0  1  0  1 -1 ]
      [ 1 -1  0 -1  0  0 ]
      [-1 -1  0 -1  0  0 ]
      [ 1  1  0 -1  0  0 ]
      [-1  1  0 -1  0  0 ]
```

Rows are motors `1..8`. Columns are `[roll, pitch, yaw, throttle, forward, lateral]`.

### 3. Motor-direction correction

ArduSub applies per-motor direction signs after raw mixing. The implemented default is:

```text
D = diag(-1, -1, +1, +1, -1, +1, +1, -1)
```

So the signed motor command used in the composite model is:

```text
u_motor = D A a_rc
```

This sign tuple is exposed as the `motor_direction_signs` argument because firmware motor
direction settings may differ from the derived default.

What a motor-direction sign means:

- `+1` means a positive raw mixer command keeps the same thrust direction as the PX4 rotor
  axis in `uuv_wrench_allocator.py`
- `-1` means a positive raw mixer command is flipped and creates thrust in the opposite
  direction

If `a_i` is the unit thrust axis from the PX4 geometry and `s_i` is the corresponding
motor-direction sign, then a positive raw motor command produces force along:

```text
f_i ∝ s_i a_i
```

For the default signs and the `60002` BlueROV2 Heavy geometry, a positive raw motor command
creates these force directions:

| Motor row | PX4 rotor axis | Sign | Positive raw command creates |
| --- | --- | ---: | --- |
| 1 | `[+X, -Y, 0]` | `-1` | backward and right |
| 2 | `[+X, +Y, 0]` | `-1` | backward and left |
| 3 | `[+X, +Y, 0]` | `+1` | forward and right |
| 4 | `[+X, -Y, 0]` | `+1` | forward and left |
| 5 | `[0, 0, -Z]` | `-1` | down |
| 6 | `[0, 0, +Z]` | `+1` | down |
| 7 | `[0, 0, +Z]` | `+1` | down |
| 8 | `[0, 0, -Z]` | `-1` | down |

The vertical result is worth calling out:

- with the default signs, a positive raw command on any vertical motor means downward force
- ArduSub's positive `throttle` axis means upward motion, so the mixer uses negative raw
  motor values for positive throttle

## Composite Translator

Substituting the ArduSub mixer into the PX4 wrench model gives:

```text
w = B D A a_rc
```

Define:

```text
C = B D A
```

Then the translator solves for RC axes using a pseudo-inverse:

```text
a_rc = pinv(C) w
```

This is the core implemented mapping.

For the default `60002` geometry and default motor-direction signs, the behavior is:

- `tau_x` maps cleanly to roll
- `tau_y` maps cleanly to pitch
- `tau_z` maps cleanly to yaw
- `F_z` maps to throttle with the expected sign flip because `+F_z` means down in `FRD`
- `F_x` maps mainly to forward, with a small pitch compensation term
- `F_y` maps mainly to lateral, with a small roll compensation term

Those small couplings are expected. They come from the PX4 geometry model accounting for
the moments generated by the horizontal thrusters.

## PWM Conversion

After computing `a_rc`, each axis is clipped to `[-1, 1]`.

Each RC axis is then converted to PWM as:

```text
PWM = neutral_pwm + delta_pwm * axis
```

with the implementation defaults:

- `neutral_pwm = 1500`
- `delta_pwm = 100`

So a fully saturated axis maps to `1400` or `1600`.

## Worked Examples

With the default implementation parameters:

- Pure roll command `tau_x = +1` produces roll channel `1517`
- Pure pitch command `tau_y = +1` produces pitch channel `1532`
- Pure yaw command `tau_z = +1` produces yaw channel `1523`
- Pure heave command `F_z = +1` produces throttle channel `1496`
  because positive `F_z` is downward force in `FRD`
- Pure surge command `F_x = +1` produces forward channel `1505` and a small pitch
  compensation `1498`
- Pure sway command `F_y = +1` produces lateral channel `1505` and a small roll
  compensation `1501`

These values are covered by tests in `test/test_brov_ctrl_wrench.py`.

## Limitations and Assumptions

- This is not a direct low-level thruster command interface.
- It assumes default ArduSub channel mapping.
- It assumes manual-mode semantics for RC override.
- It assumes the incoming wrench magnitudes are already meaningful in the modeled
  allocator space.
- It does not model ArduSub's group-wise normalization, trims, current limiting, or
  assisted-mode controllers.

## References

- `wrench_controller_design.md`
- `bodycmd_mixer.md`
- `uuv_wrench_allocator.py`
- `brov_ctrl_wrench.py`
