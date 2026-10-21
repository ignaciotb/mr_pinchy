"""Math for the relative-state safety filter (no ROS in this file).

Everything is 2D and expressed in the robot body frame (FLU: x forward, y left).
The environment is assumed static: every sonar point is a fixed obstacle.
See doc/relative_safety_filter.md for the derivation.

  P      (N, 2) obstacle points relative to the robot
  u      robot velocity command [vx, vy]
"""
from __future__ import annotations

import math

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp


def rotation_2d(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def _softmin_h(q: jnp.ndarray, points: jnp.ndarray, d_safe: float, kappa: float) -> jnp.ndarray:
    """h(q) = -(1/kappa) log sum_i exp(-kappa h_i),  h_i = |p_i - q| - d_safe.
    A smooth lower bound on min_i h_i (within log(N)/kappa of it)."""
    h_i = jnp.sqrt(jnp.sum((points - q) ** 2, axis=1) + 1e-12) - d_safe
    return -logsumexp(-kappa * h_i) / kappa


_h_and_grad = jax.jit(jax.value_and_grad(_softmin_h))


class SoftminCbf:
    """Soft-min barrier over all obstacle points, evaluated with JAX.

    With q the robot position (origin of the body frame) and static points, the robot
    moves through the points at velocity u, so  h_dot = grad_q h · u  (the yaw rate drops
    out because grad_q h is a combination of radial directions, see doc).
    The CBF condition h_dot >= -alpha*h is the half-plane  n·u <= b  with
        n = -grad_q h   (points toward the obstacles),   b = alpha*h.

    The point array is padded / cut to max_points so JAX compiles once."""

    def __init__(self, d_safe: float, alpha: float, kappa: float, max_points: int = 2048) -> None:
        self.d_safe, self.alpha, self.kappa, self.max_points = d_safe, alpha, kappa, max_points
        self.constraint(np.array([[1e3, 0.0]]))  # warm up the JIT

    def constraint(self, points_xy: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Returns (n, b, h) for the half-plane n·u <= b."""
        pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
        dist2 = np.sum(pts**2, axis=1)
        if pts.shape[0] > self.max_points:  # keep the nearest ones
            pts = pts[np.argpartition(dist2, self.max_points)[: self.max_points]]
        # Padding points are far away, so exp(-kappa*h_i) underflows to exactly 0.
        pad = np.full((self.max_points - pts.shape[0], 2), 1e3, dtype=np.float32)
        h, grad = _h_and_grad(jnp.zeros(2), jnp.asarray(np.vstack([pts, pad])), self.d_safe, self.kappa)
        n = -np.asarray(grad, dtype=float)
        h = float(h)
        return n, self.alpha * h, h


class CbfQp:
    """min 0.5|u - u_ref|^2 + 0.5 w_u |u - u_prev|^2 + w_s*s
    s.t.  n·u <= b + s,  s >= 0,  |u_i| <= v_max.

    The u_prev term penalizes tick-to-tick changes of the filtered command, which damps
    the jitter caused by the constraint switching with the noisy sonar cloud.
    smooth_weight = 0 recovers the plain CBF-QP.

    The problem is built once; solve() only updates the parameters."""

    def __init__(self, v_max: float, slack_weight: float, smooth_weight: float = 0.0) -> None:
        self.u = cp.Variable(2)
        self.s = cp.Variable(nonneg=True)
        self.u_ref = cp.Parameter(2)
        self.u_prev = cp.Parameter(2, value=np.zeros(2))
        self.n = cp.Parameter(2)
        self.b = cp.Parameter()
        cost = (
            0.5 * cp.sum_squares(self.u - self.u_ref)
            + 0.5 * smooth_weight * cp.sum_squares(self.u - self.u_prev)
            + slack_weight * self.s
        )
        constraints = [self.n @ self.u <= self.b + self.s, cp.abs(self.u) <= v_max]
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(
        self, u_ref: np.ndarray, n: np.ndarray, b: float, u_prev: np.ndarray | None = None
    ) -> tuple[np.ndarray | None, float]:
        self.u_ref.value = np.asarray(u_ref, dtype=float)
        self.u_prev.value = np.zeros(2) if u_prev is None else np.asarray(u_prev, dtype=float)
        self.n.value = np.asarray(n, dtype=float)
        self.b.value = float(b)
        # OSQP first (fast, warm-started); CLARABEL as a fallback if OSQP is not clean.
        for solver in (cp.OSQP, cp.CLARABEL):
            try:
                self.problem.solve(solver=solver, warm_start=True)
            except cp.SolverError:
                continue
            if self.problem.status == cp.OPTIMAL and self.u.value is not None:
                return np.asarray(self.u.value, dtype=float), float(self.s.value)
        return None, 0.0
