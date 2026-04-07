# How ArduPilot Digests `/mavros/rc/override` for BlueROV2 Heavy

This note documents the ArduPilot side of the path from ROS `/mavros/rc/override` to per-thruster output for ArduSub with the `VECTORED_6DOF` frame, which is the BlueROV2 Heavy mixer.

At the MAVLink layer, `/mavros/rc/override` becomes `RC_CHANNELS_OVERRIDE`. ArduPilot does not treat this as a body-wrench message. It treats it as synthetic RC stick input, then runs the normal ArduSub control and mixing pipeline.

## 1. Message Path

The relevant path in ArduPilot is:

1. MAVROS publishes `/mavros/rc/override`.
2. ArduPilot receives `MAVLINK_MSG_ID_RC_CHANNELS_OVERRIDE`.
3. `GCS_MAVLINK_Sub::handle_message()` accepts it and marks pilot input as alive.
4. `GCS_MAVLINK::handle_rc_channels_override()` writes the override PWM values into the RC channel objects.
5. ArduSub flight mode code reads those RC channels as if they came from a transmitter.
6. The motors library mixes the resulting roll, pitch, yaw, throttle, forward, and lateral demands into motor commands.

Code references:

- [`ArduSub/GCS_MAVLink_Sub.cpp`](ArduSub/GCS_MAVLink_Sub.cpp): `RC_CHANNELS_OVERRIDE` is accepted in `handle_message()`.
- [`libraries/GCS_MAVLink/GCS_Common.cpp`](libraries/GCS_MAVLink/GCS_Common.cpp): `handle_rc_channels_override()` writes override values into `RC_Channels`.
- [`libraries/RC_Channel/RC_Channels.cpp`](libraries/RC_Channel/RC_Channels.cpp): `RC_Channels::set_override()`.

## 2. Which Override Channel Means What

ArduSub resolves semantic axes through `RCMAP_*`, not by hard-coding channel meaning directly in the mixer.

The input channels are bound in [`ArduSub/radio.cpp`](ArduSub/radio.cpp):

- `channel_roll     = rc().get_roll_channel()`
- `channel_pitch    = rc().get_pitch_channel()`
- `channel_throttle = rc().get_throttle_channel()`
- `channel_yaw      = rc().get_yaw_channel()`
- `channel_forward  = rc().get_forward_channel()`
- `channel_lateral  = rc().get_lateral_channel()`

Those accessors resolve through the RC mapper in [`libraries/RC_Channel/RC_Channels.cpp`](libraries/RC_Channel/RC_Channels.cpp).

### Default ArduSub mapping

By default, ArduSub changes some of the generic RC mapper defaults in [`ArduSub/Parameters.h`](ArduSub/Parameters.h):

| Semantic axis | Default RC map |
| --- | --- |
| Pitch | `RCMAP_PITCH = 1` |
| Roll | `RCMAP_ROLL = 2` |
| Throttle | `RCMAP_THROTTLE = 3` |
| Yaw | `RCMAP_YAW = 4` |
| Forward | `RCMAP_FORWARD = 5` |
| Lateral | `RCMAP_LATERAL = 6` |

So, with default ArduSub settings, `/mavros/rc/override.channels` maps as:

| Override index | MAVLink channel | ArduSub meaning |
| --- | --- | --- |
| `channels[0]` | RC1 | pitch |
| `channels[1]` | RC2 | roll |
| `channels[2]` | RC3 | throttle |
| `channels[3]` | RC4 | yaw |
| `channels[4]` | RC5 | forward |
| `channels[5]` | RC6 | lateral |

This is only the default. If `RCMAP_*` parameters are changed, the meaning of each override channel changes too.

## 3. Normalization Inside ArduPilot

After the PWM overrides are stored, ArduSub reads them through `RC_Channel`.

For roll, pitch, yaw, forward, and lateral, ArduSub typically uses:

```text
norm_input() in [-1, 1]
```

For throttle, ArduSub first maps the channel to:

```text
(channel_throttle->norm_input() + 1) / 2  in [0, 1]
```

Then the motors layer converts that back to bidirectional thrust with:

```text
2 * (filtered_throttle - 0.5)  in [-1, 1]
```

Relevant code:

- [`libraries/RC_Channel/RC_Channel.cpp`](libraries/RC_Channel/RC_Channel.cpp): `RC_Channel::norm_input()`
- [`libraries/AP_Motors/AP_Motors_Class.h`](libraries/AP_Motors/AP_Motors_Class.h): `set_roll()`, `set_pitch()`, `set_yaw()`, `set_throttle()`, `set_forward()`, `set_lateral()`, and `get_throttle_bidirectional()`

In practice, centered PWM around trim gives approximately zero command, low gives negative command, and high gives positive command.

## 4. Mode Matters

`RC_CHANNELS_OVERRIDE` does not always become a direct body-axis thrust command.

### Manual mode

In `MANUAL`, the override is almost direct stick-to-mixer input:

- roll -> `motors.set_roll(channel_roll->norm_input())`
- pitch -> `motors.set_pitch(channel_pitch->norm_input())`
- yaw -> `motors.set_yaw(channel_yaw->norm_input() * gain)`
- throttle -> `motors.set_throttle((norm_input + 1) / 2)`
- forward -> `motors.set_forward(channel_forward->norm_input())`
- lateral -> `motors.set_lateral(channel_lateral->norm_input())`

See [`ArduSub/mode_manual.cpp`](ArduSub/mode_manual.cpp).

This is the closest mode to "body-axis command to thruster command".

### Stabilize / depth-hold / other assisted modes

In assisted modes, roll, pitch, and often yaw are not fed directly to the mixer. They first go through the attitude controller, which generates the actual motor roll/pitch/yaw demands. Forward and lateral remain pilot translational inputs.

See [`ArduSub/mode_stabilize.cpp`](ArduSub/mode_stabilize.cpp).

So if you publish the same `/mavros/rc/override` message in different modes, the motor outputs can be very different.

## 5. Axis Direction Convention

`RC_CHANNELS_OVERRIDE` itself has no frame field, but ArduSub interprets the six pilot axes with a consistent body-axis sign convention.

### Translational axes

For body-frame MAVLink setpoints, ArduSub accepts `MAV_FRAME_BODY_NED`, `MAV_FRAME_BODY_OFFSET_NED`, and `MAV_FRAME_BODY_FRD`, then converts:

```text
x -> forward
y -> right
z -> down   (MAVLink body frame)
```

into ArduSub's internal position/velocity convention by flipping `z`:

```text
internal x -> forward
internal y -> right
internal z -> up
```

This is visible in [`ArduSub/GCS_MAVLink_Sub.cpp`](ArduSub/GCS_MAVLink_Sub.cpp), where ArduSub forms:

```text
Vector3f(packet.x, packet.y, -packet.z)
```

for body-frame position and velocity targets before rotating `x/y` into NE coordinates.

For the RC-style axes used by `/mavros/rc/override`, the signs line up with that same body convention:

- positive `forward` means forward surge
- positive `lateral` means rightward sway
- positive `throttle` means upward heave demand

The upward interpretation of positive throttle is explicit in the climb-rate code in [`ArduSub/Attitude.cpp`](ArduSub/Attitude.cpp): values above the throttle deadband produce positive `pilot_speed_up`, while values below it produce negative descent-rate demand.

### Angular axes

The angular signs are:

- positive `roll` means roll right
- positive `pitch` means pitch up
- positive `yaw` means yaw right, i.e. clockwise when viewed from above

Evidence in the code:

- Yaw: [`ArduSub/turn_counter.cpp`](ArduSub/turn_counter.cpp) documents that yaw moving from negative to positive is a "right turn".
- Roll/lateral linkage: [`ArduSub/motors.cpp`](ArduSub/motors.cpp) maps positive roll controller output to positive lateral demand, so positive roll corresponds to rightward tilt/right-side-down roll.
- Pitch/forward linkage: [`ArduSub/motors.cpp`](ArduSub/motors.cpp) uses `forward = -pitch`, so positive forward motion corresponds to negative pitch. Equivalently, positive pitch is nose-up and negative pitch is nose-down/forward-driving.

### Important mixer sign caveat

Vehicle-axis sign and per-motor sign are not the same thing.

For example, in the BlueROV2 Heavy mixer the vertical thrusters have `Throttle = -1`, so a positive upward heave command contributes a negative raw motor command on motors `5..8` before `MOT_n_DIRECTION` is applied. That negative sign is a property of thruster installation and mixer coefficients, not a contradiction of the vehicle-axis convention.

## 6. BlueROV2 Heavy Mixer: `VECTORED_6DOF`

For BlueROV2 Heavy, the motor frame is `SUB_FRAME_VECTORED_6DOF` in [`libraries/AP_Motors/AP_Motors6DOF.cpp`](libraries/AP_Motors/AP_Motors6DOF.cpp).

The per-motor factors are:

| Motor | Roll | Pitch | Yaw | Throttle | Forward | Lateral |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 1 | 0 | -1 | 1 |
| 2 | 0 | 0 | -1 | 0 | -1 | -1 |
| 3 | 0 | 0 | -1 | 0 | 1 | 1 |
| 4 | 0 | 0 | 1 | 0 | 1 | -1 |
| 5 | 1 | -1 | 0 | -1 | 0 | 0 |
| 6 | -1 | -1 | 0 | -1 | 0 | 0 |
| 7 | 1 | 1 | 0 | -1 | 0 | 0 |
| 8 | -1 | 1 | 0 | -1 | 0 | 0 |

See [`libraries/AP_Motors/AP_Motors6DOF.cpp`](libraries/AP_Motors/AP_Motors6DOF.cpp), case `SUB_FRAME_VECTORED_6DOF`.

## 7. The Actual Mixing Equations

For `VECTORED_6DOF`, ArduPilot uses a special mixing function:

- `output_armed_stabilizing_vectored_6dof()`

in [`libraries/AP_Motors/AP_Motors6DOF.cpp`](libraries/AP_Motors/AP_Motors6DOF.cpp).

It splits the vehicle into two groups:

- Horizontal thrusters, motors `1..4`: use `yaw + forward + lateral`
- Vertical thrusters, motors `5..8`: use `roll + pitch + throttle`

Let:

```text
r = roll command
p = pitch command
y = yaw command
t = bidirectional throttle command
f = forward command
l = lateral command
```

Then the raw group outputs are:

```text
m1_raw =  y - f + l
m2_raw = -y - f - l
m3_raw = -y + f + l
m4_raw =  y + f - l

m5_raw =  r - p - t
m6_raw = -r - p - t
m7_raw =  r + p - t
m8_raw = -r + p - t
```

ArduPilot then computes:

```text
yfl_max = max(1, abs(m1_raw), abs(m2_raw), abs(m3_raw), abs(m4_raw))
rpt_max = max(1, abs(m5_raw), abs(m6_raw), abs(m7_raw), abs(m8_raw))
```

and normalizes each group independently:

```text
m1 = m1_raw / yfl_max
m2 = m2_raw / yfl_max
m3 = m3_raw / yfl_max
m4 = m4_raw / yfl_max

m5 = m5_raw / rpt_max
m6 = m6_raw / rpt_max
m7 = m7_raw / rpt_max
m8 = m8_raw / rpt_max
```

This independent normalization is specific to the `VECTORED_6DOF` path. It prevents one group from saturating the other group.

Finally, each motor output is:

```text
motor_i = constrain(MOT_i_DIRECTION * m_i, -1, 1)
```

and then converted to PWM around 1500:

```text
PWM = 1500 + thrust * range
```

See:

- [`libraries/AP_Motors/AP_Motors6DOF.cpp`](libraries/AP_Motors/AP_Motors6DOF.cpp): `output_armed_stabilizing_vectored_6dof()`
- [`libraries/AP_Motors/AP_Motors6DOF.cpp`](libraries/AP_Motors/AP_Motors6DOF.cpp): `calc_thrust_to_pwm()`

## 8. What This Means for `/mavros/rc/override`

If you publish `/mavros/rc/override` with default ArduSub channel mapping on BlueROV2 Heavy:

- pitch command affects only motors `5..8`
- roll command affects only motors `5..8`
- throttle command affects only motors `5..8`
- yaw command affects only motors `1..4`
- forward command affects only motors `1..4`
- lateral command affects only motors `1..4`

So the Heavy frame is effectively block-diagonal:

- vertical thrusters control `roll/pitch/heave`
- horizontal thrusters control `yaw/surge/sway`

This is why `VECTORED_6DOF` is easy to reason about as two separate 4-thruster sub-mixers.

## 9. Important Caveats

- `RC_CHANNELS_OVERRIDE` is interpreted as RC input, not as a low-level thruster command.
- The exact meaning of override channel indices depends on `RCMAP_*`.
- The semantic axis convention is body-style: forward/right/up for translation and roll-right, pitch-up, yaw-right for rotation.
- The exact motor response depends on flight mode.
- `MOT_n_DIRECTION` can flip signs after the nominal mixer.
- PWM output range depends on the configured RC min/max for the throttle channel and motor output setup.
- Battery current limiting can scale motor outputs after mixing.

## 10. Summary

For BlueROV2 Heavy in `MANUAL` mode, `/mavros/rc/override` is close to a body-axis command interface:

- RC override channels become normalized pilot axes.
- Those axes are written into the ArduSub motors object.
- `AP_Motors6DOF` applies the `VECTORED_6DOF` coefficient table.
- Horizontal thrusters mix `yaw/forward/lateral`.
- Vertical thrusters mix `roll/pitch/throttle`.
- The result is normalized, direction-adjusted, and converted to PWM per motor.

In assisted modes, the same override input is not a direct body-axis command anymore because attitude and depth controllers sit between RC input and motor mixing.
