# Low-level Controller Tuning

**Files:**
- `brov_ctrl_twist.py` — inner loop (velocity → wrench → RC PWM)
- `brov_ctrl_waypoint_twist.py` — outer loop (position → velocity command)

**Observed issues:**
- Z-axis waypoint stabilization oscillates
- Yaw rate cannot follow reference; oscillates when gain is increased
- Control frequency is low (30 Hz)
- Certain commands cause large PWM jumps

---

## Root Causes

**Z-axis oscillation:** Zero derivative gain in both loops (`_kd_w=0`, `pid_z.kd=0`). Asymmetric heave P-gain (10 up / 20 down) creates different dynamics per direction.

**Yaw rate tracking failure:** The dynamics model has `Nr=0` (zero linear yaw damping), but the real vehicle has significant damping. Feedforward undercompensates, forcing P-gain to do all the work. With P-only feedback, steady-state error = `Nr_real·r / (k_r + Nr_real)` — never reaches zero. Increasing `k_r` hits the oscillation limit before closing the gap. **Fix: add integral action** to reject unmodeled damping as a constant disturbance.

**Low control frequency:** 30 Hz limits derivative quality and reduces the max stable gain.

**Large PWM jumps:** `delta_pwm=300` with no rate limiting. Feedforward wrench changes abruptly when velocity changes (quadratic damping).

---

## Plan

Tune inner loop first, then outer loop. Test one change at a time.

### Phase 0: Diagnostics

Add publishers in both controllers for step-response visualization:

- [ ] `brov_ctrl_twist.py`: publish `nu_ref`, `nu`, velocity error, `tau_ff`, `tau_fb`, `tau`, RC PWM (use `Float64MultiArray` topics under `/mr_pinchy/ctrl_diag/`)
- [ ] `brov_ctrl_waypoint_twist.py`: publish position error `[ex,ey,ez]`, orientation error `[eroll,epitch,eyaw]`, PID outputs
- [ ] Set up PlotJuggler layout: `nu_ref` vs `nu` per axis, wrench breakdown, PWM channels, position error

### Phase 1: Inner Loop (`brov_ctrl_twist.py`)

Send constant velocity commands via `ros2 topic pub` to bypass the waypoint controller.

#### 1a. Increase control rate

- [ ] Check odom rate: `ros2 topic hz /mocap/mr_pinchy/odom_ekf`
- [ ] Raise `_publish_rate_hz` from 30 to 50–100 (must not exceed odom rate)

#### 1b. Fix heave oscillation

- [ ] Add heave derivative gain: `_kd_w = 2.0–5.0` (currently 0)
- [ ] Make heave P-gain symmetric: `_k_w_pos = _k_w_neg = 15.0` (currently 10/20)
- [ ] If still oscillating, reduce heave P-gain to ~8.0 (feedforward handles steady-state damping)

#### 1c. Fix yaw rate tracking

P-gain alone can't eliminate steady-state error from unmodeled damping — integral action is needed.

- [ ] Add integral state for yaw rate in `BrovTwistController`:
  - Keep `_k_r = 4.0` (don't increase — causes oscillation)
  - Add `_ki_r = 2.0–4.0` with anti-windup (clamp integral to ±5.0 Nm)
  - Keep `_kd_r = 0.1` initially
- [ ] Increase reference filter alpha for yaw from 0.1 to 0.3–0.5 (make per-axis so heave is unaffected)
- [ ] Optional: identify real `Nr` from step-response data and update `brov_dynamics.py` to fix the feedforward

#### 1d. Limit PWM rate of change

- [ ] Add per-tick PWM rate limiting in `_publish_wrench_and_rc()` (e.g. ±30 PWM/tick)
- [ ] Optionally reduce `_delta_pwm` from 300 to 200

### Phase 2: Outer Loop (`brov_ctrl_waypoint_twist.py`)

Only after inner loop tracks velocity commands cleanly.

#### 2a. Fix z-axis waypoint oscillation

- [ ] Add `pid_z.kd = 0.3–0.5` (decelerates before overshoot)
- [ ] Add `pid_z.ki = 0.05–0.1` (eliminates steady-state depth error from buoyancy)

#### 2b. Speed up yaw waypoint response

- [ ] Increase `angular_vel_limit` from 0.15 to 0.3–0.5 rad/s (primary bottleneck)
- [ ] Increase `_pid_yaw.kp` from 3.0 to 4.0–5.0 (only after velocity limit is raised)

#### 2c. XY position (if needed)

- [ ] Add `kd = 0.2–0.3` to `_pid_x` / `_pid_y` if overshoot is observed

### Phase 3: Integration Test

- [ ] Run waypoint tracking (square / figure-8 trajectory)
- [ ] Verify: Z settles < 2s with < 5% overshoot, yaw settles < 1.5s for 90°, no PWM jump > 100/tick, XY error < 0.1m at cruise
- [ ] Record rosbag for post-analysis

---

## Parameter Reference

| Parameter | Current | Suggested | File | Line |
|---|---|---|---|---|
| `_publish_rate_hz` | 30 | 50–100 | brov_ctrl_twist.py | 89 |
| `_k_w_pos` | 10 | 15 | brov_ctrl_twist.py | 94 |
| `_k_w_neg` | 20 | 15 | brov_ctrl_twist.py | 95 |
| `_kd_w` | 0.0 | 2.0–5.0 | brov_ctrl_twist.py | 101 |
| `_k_r` | 4.0 | 4.0 (keep) | brov_ctrl_twist.py | 98 |
| `_ki_r` | — | 2.0–4.0 (new) | brov_ctrl_twist.py | — |
| `_kd_r` | 0.1 | 0.1 (keep) | brov_ctrl_twist.py | 104 |
| `alpha` | 0.1 | 0.3–0.5 (yaw) | brov_ctrl_twist.py | 409 |
| `_delta_pwm` | 300 | 200–300 | brov_ctrl_twist.py | 112 |
| `pid_z.kd` | 0.0 | 0.3–0.5 | brov_ctrl_waypoint_twist.py | 44 |
| `pid_z.ki` | 0.0 | 0.05–0.1 | brov_ctrl_waypoint_twist.py | 44 |
| `angular_vel_limit` | 0.15 | 0.3–0.5 | brov_ctrl_waypoint_twist.py | 48 |
| `_pid_yaw.kp` | 3.0 | 4.0–5.0 | brov_ctrl_waypoint_twist.py | 64 |
