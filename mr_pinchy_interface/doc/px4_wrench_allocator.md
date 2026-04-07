# px4_wrench_allocator.md

## Axis Convention

`Tools/uuv_wrench_allocator.py` follows the same body-axis convention used by the PX4 control allocator:

- Body frame is `FRD`.
- `+X`: forward.
- `+Y`: right / starboard.
- `+Z`: down.
- Positive torque follows the right-hand rule about each positive body axis.

So the wrench/control vector is ordered as:

```text
w = [tau_x, tau_y, tau_z, F_x, F_y, F_z]
    [ roll,  pitch, yaw,  surge, sway, heave ]
```

Important sign consequence:

- `+F_z` means force down, not up.
- `-F_z` means force up.
- With `+Z` down, positive yaw (`+tau_z`) is clockwise when viewed from above.

## BlueROV2 Heavy Thruster Convention

The current BlueROV2 Heavy airframe in PX4 is `60002_uuv_bluerov2_heavy`.
Its rotor parameters are copied directly into `uuv_wrench_allocator.py`.

Interpretation of each rotor entry:

```text
(axis_x, axis_y, axis_z, pos_x, pos_y, pos_z, km)
```

- `axis_*` is the thrust direction for a positive motor command.
- `pos_*` is the rotor position in the same body `FRD` frame.
- `km` is the propeller reaction-torque ratio. It is zero for this BlueROV2 Heavy setup.
- `CA_R_REV = 255`, so all thrusters are reversible. A negative motor command generates thrust in the opposite direction.

For the `60002` geometry:

- Rotors `0..3` are horizontal thrusters used mainly for surge, sway, and yaw.
- Rotors `4..7` are vertical thrusters used mainly for heave, roll, and pitch.
- Rotors `4` and `7` have `axis_z = -1`, so a positive command produces upward force (`-F_z`).
- Rotors `5` and `6` have `axis_z = +1`, so a positive command produces downward force (`+F_z`).

## How Wrench Becomes Motor Input

PX4 builds one column per thruster and stacks them into the effectiveness matrix `B`.

For thruster `i`:

```text
a_i = normalized thrust axis
p_i = thruster position

force_i  = ct_i * a_i
moment_i = ct_i * (p_i x a_i) - ct_i * km_i * a_i
```

Then:

```text
          [ moment_0  moment_1 ... moment_7 ]
B =       [  force_0   force_1 ...  force_7 ]

w = B u
```

where:

- `w` is the desired body wrench/control vector.
- `u` is the 8-element motor command vector.

Because the vehicle is over-actuated (`8` thrusters for `6` axes), PX4 solves the inverse problem with a pseudo-inverse:

```text
u ~= pinv(B) w
```

Then PX4 normalizes that pseudo-inverse into a mixer matrix and clips outputs to actuator limits:

```text
mix = normalize(pinv(B))
u   = clip(mix @ w, -1, 1)
```

That is the path used by the current `60002` BlueROV2 Heavy airframe because it sets `CA_METHOD = 0`.

## Why Some Axes Use "Unexpected" Thrusters

A pure translational request does not always mean "only the obvious thrusters".

Example: a positive surge command uses the four horizontal thrusters, but those thrusters are mounted above the vehicle center (`p_z > 0`), so they also create roll/pitch moments through `p x f`.
The allocator therefore adds small vertical-thruster commands to cancel those unwanted moments while still achieving the requested net force.

That is expected behavior, not a sign error.

## Quick Sign Checks

Useful mental checks for the `60002` geometry:

- `+F_z` (down): rotors `5` and `6` positive, rotors `4` and `7` negative.
- `+tau_z` (yaw): horizontal thrusters follow the sign pattern `[-, +, -, +]`.
- `+tau_x` (roll): vertical thrusters follow the sign pattern `[-, -, +, +]`.
- `+tau_y` (pitch): vertical thrusters follow the sign pattern `[+, -, +, -]`.

## Correctness Check on `uuv_wrench_allocator.py`

What is correct:

- The rotor geometry matches the PX4 BlueROV2 Heavy airframe files for `60002` and `1022`.
- The effectiveness-matrix construction matches `ActuatorEffectivenessRotors.cpp`.
- The control-vector ordering matches `ControlAllocator.cpp`.
- The axis convention is body `FRD`, not `FLU` and not `ENU`.

Main caveat:

- The current PX4 `60002` airframe uses pseudo-inverse allocation with output clipping, so this script is a good match for that case.
- The older `1022` setup does not explicitly set `CA_METHOD = 0`; if left in automatic mode, PX4 may instead use sequential desaturation for UUVs.

Most likely source of confusion:

- If you expected underwater "heave up" to be positive, this script will look flipped. In PX4 `FRD`, positive heave is down because `+Z` is down.
