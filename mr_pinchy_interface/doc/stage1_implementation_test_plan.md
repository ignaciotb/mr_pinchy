# Stage 1 — Implementation & Test Plan

**Scope.** Inner-loop velocity tracker + outer-loop kinematic CBF-QP on the
von Benzon et al. (2022) BlueROV2 simulator, under idealized assumptions
(no disturbance, no current, no tether, perfect model, perfect measurements).

---

## 1. Control law

**Inner loop (inverse dynamics + PD), run at ~200 Hz:**
$$\tau = M\dot\nu_d + C(\nu)\nu + D(\nu)\nu + g(\eta) + K_\nu(\nu_d - \nu)$$

**Outer loop (kinematic CBF-QP), run at ~50 Hz:**
$$
\nu_d = \arg\min_{\nu}\ \|\nu - \nu_{\text{nom}}\|^2
\quad\text{s.t.}\quad
\nabla h(\eta)^\top J(\eta)\,\nu + \gamma\,h(\eta) \geq 0,\ \nu_{\min} \leq \nu \leq \nu_{\max}
$$

**Closed-loop guarantee (Stage 1 only):** $M\dot{\tilde\nu} = -K_\nu\tilde\nu$,
so $\nu \to \nu_d$ exponentially and the outer plant is exactly $\dot\eta = J(\eta)\nu_d$.

---

## 2. Implementation steps

1. **Plant.** Use the von Benzon et al. Simulink simulator unmodified.
   Disable the tether block, ocean current block, and sensor noise block.
2. **Parameters.** Load $M, C, D, g$ from Table A1. Pre-compute $M$ as a
   constant matrix; compute $C(\nu)$ and $D(\nu)$ symbolically per timestep.
3. **Inner-loop block.** Inputs: $\nu_d, \dot\nu_d, \nu, \eta$. Output: $\tau$.
   Implement Eq. above directly. Initial gains:
   - $K_\nu = \mathrm{diag}(50, 50, 50, 5, 5, 5)$ (translational / rotational).
4. **Trajectory tracker (provides $\nu_{\text{nom}}$).** Simple proportional law:
   $\nu_{\text{nom}} = J(\eta)^{-1}(\dot\eta_r + K_p(\eta_r - \eta))$ with
   $K_p = \mathrm{diag}(1, 1, 1, 0.5, 0.5, 0.5)$.
5. **Barrier function.** Single static spherical obstacle:
   $h(\eta) = \|\eta_{xyz} - \eta_{\text{obs}}\|^2 - r^2$.
   Place $\eta_{\text{obs}}$ between start and goal so the QP is forced to act.
6. **CBF-QP.** Use `quadprog` (MATLAB) or OSQP. 6 decision variables, 1 CBF
   constraint, 12 box constraints. Class-$\mathcal{K}$ gain $\gamma = 1.0$.
7. **$\dot\nu_d$.** For Stage 1, set $\dot\nu_d = 0$ in the inner loop. The PD
   term will absorb the small lag (Stage 2 adds the proper smoother).
8. **Thruster allocation.** Use the existing $\tau \to V$ block from the
   simulator (Eq. 13 of von Benzon et al.). Disable saturation for Stage 1.

---

## 3. Test cases

Run each in the order listed. Each test isolates one capability.

| # | Test | Setup | Pass criterion |
|---|---|---|---|
| **T1** | Inner loop step response | Hold $\eta_r$ fixed, command step in $\nu_d$ (no CBF, no obstacle) | $\|\tilde\nu\| < 1\%$ of step magnitude within 5 inner-loop time constants |
| **T2** | Inner loop tracking | Sinusoidal $\nu_d(t)$, no CBF | RMS tracking error < 2% per DOF |
| **T3** | Outer loop, no obstacle | Trajectory tracker only, CBF inactive ($h \gg 0$ everywhere) | $\|\eta - \eta_r\| < 5$ cm steady-state |
| **T4** | CBF activation | Straight-line trajectory through obstacle center | $h(\eta(t)) \geq 0\ \forall t$; deviation only near obstacle |
| **T5** | CBF tangential slide | Trajectory grazes obstacle | Smooth slide along $h = 0$ boundary, no chattering |
| **T6** | Multiple goals | Sequence of waypoints around obstacle | All waypoints reached, $h \geq 0$ throughout |

---

## 4. Metrics to log

For every run, save time series of:

- $\eta(t), \nu(t), \nu_d(t), \tilde\nu(t)$
- $\tau(t)$ and per-thruster forces (check we're inside Table 4 limits even though saturation is off)
- $h(\eta(t))$ — must stay nonnegative
- CBF constraint slack $\nabla h^\top J\nu_d + \gamma h$ — should be $\geq 0$ and tight when active
- QP solve time per step

---

## 5. Expected failure modes (and what they mean)

| Symptom | Likely cause | Fix |
|---|---|---|
| $\tilde\nu$ doesn't decay | Wrong sign in $K_\nu$ feedback, or $C, D$ evaluated at wrong velocity | Check derivation, log $M\dot{\tilde\nu} + K_\nu\tilde\nu$ — should be $\approx 0$ |
| $h$ goes slightly negative | Inner loop too slow → reduced-order abstraction inaccurate | Increase $K_\nu$, or move to Stage 2 |
| QP infeasible | Box constraints conflict with CBF constraint | Loosen $\gamma$, or check obstacle geometry |
| Chattering near $h=0$ | $\gamma$ too large, or numerical noise | Lower $\gamma$, add small slack on the CBF constraint |
| $\tau$ exceeds Table 4 limits | $K_\nu$ too large, or $\nu_{\text{nom}}$ too aggressive | Lower gains; this is the cue that Stage 4 actuator headroom matters |

---

## 6. Exit criteria for Stage 1

Move on to Stage 2 when **all** of the following hold on the noise-free,
disturbance-free simulator:

- T1–T6 pass.
- Forward invariance of $\{h \geq 0\}$ holds across at least 10 randomized
  start/goal/obstacle configurations.
- Inner-loop bandwidth is verifiably ≥5× outer-loop bandwidth (measured from
  step responses in T1 and T3).
- QP solve time < 5 ms per step on the target machine.
