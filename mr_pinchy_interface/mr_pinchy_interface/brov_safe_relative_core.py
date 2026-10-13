"""Math for the relative-state safety filter (no ROS in this file).

Everything is 2D and expressed in the robot body frame (FLU: x forward, y left).
See doc/relative_safety_filter.md for the derivation.

  p      obstacle position relative to the robot
  v_obs  obstacle velocity, expressed in the body frame
  v_r    robot velocity (DVL)
  omega  robot yaw rate (gyro)
  u      robot velocity command [vx, vy]
"""
from __future__ import annotations

import math

import cvxpy as cp
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def rotation_2d(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


class RelativeKF:
    """Linear Kalman filter on x = [p, v_obs] (4 states).

    Prediction over dt while the robot moves with v_r and turns with omega:
        p⁺     = R(-omega*dt) p     + dt * (v_obs - v_r)
        v_obs⁺ = R(-omega*dt) v_obs
    i.e. the body frame rotates under a world-constant obstacle velocity.
    Measurement: z = p (sonar cluster centroid).
    """

    def __init__(self, q_pos: float, q_vel: float, r_meas: float) -> None:
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])  # per second
        self.R = (r_meas**2) * np.eye(2)
        self.H = np.hstack([np.eye(2), np.zeros((2, 2))])
        self.x = np.zeros(4)
        self.P = np.eye(4)

    def reset(self, z: np.ndarray) -> None:
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([self.R[0, 0], self.R[1, 1], 1.0, 1.0])

    def predict(self, dt: float, v_robot: np.ndarray, omega: float) -> None:
        if dt <= 0.0:
            return
        rot = rotation_2d(-omega * dt)
        A = np.block([[rot, dt * np.eye(2)], [np.zeros((2, 2)), rot]])
        self.x = A @ self.x + np.concatenate([-dt * np.asarray(v_robot), [0.0, 0.0]])
        self.P = A @ self.P @ A.T + self.Q * dt

    def update(self, z: np.ndarray) -> None:
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z) - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def p(self) -> np.ndarray:
        return self.x[:2]

    @property
    def v_obs(self) -> np.ndarray:
        return self.x[2:]


def nearest_cluster_centroid(
    points_xy: np.ndarray, eps: float, min_pts: int, max_points: int = 400
) -> np.ndarray | None:
    """Single-linkage clustering (same recipe as sonar_filter); returns the centroid
    of the cluster closest to the robot that has at least min_pts points."""
    points_xy = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    n = points_xy.shape[0]
    if n < min_pts:
        return None
    if n > max_points:  # linkage is O(n^2); keep it cheap
        idx = np.random.default_rng(0).choice(n, max_points, replace=False)
        points_xy = points_xy[idx]
    labels = fcluster(linkage(points_xy, method="single"), t=eps, criterion="distance")

    best, best_dist = None, math.inf
    for label in np.unique(labels):
        cluster = points_xy[labels == label]
        if cluster.shape[0] < min_pts:
            continue
        centroid = cluster.mean(axis=0)
        dist = float(np.linalg.norm(centroid))
        if dist < best_dist:
            best, best_dist = centroid, dist
    return best


def cbf_constraint(
    p: np.ndarray, v_obs: np.ndarray, d_safe: float, alpha: float
) -> tuple[np.ndarray, float, float]:
    """h = |p| - d_safe, n = p/|p|.  Because the relative dynamics are
    p_dot = v_obs - u - omega*J*p and p·(J p) = 0, the yaw rate drops out:
        h_dot = n·(v_obs - u)
    The CBF condition h_dot >= -alpha*h becomes the half-plane  n·u <= b  with
        b = n·v_obs + alpha*h.
    Returns (n, b, h)."""
    dist = float(np.linalg.norm(p))
    if dist < 1e-6:
        n = np.array([1.0, 0.0])
    else:
        n = np.asarray(p) / dist
    h = dist - d_safe
    b = float(n @ v_obs) + alpha * h
    return n, b, h


class CbfQp:
    """min 0.5|u - u_ref|^2 + w_s*s   s.t.  n·u <= b + s,  s >= 0,  |u_i| <= v_max.

    The problem is built once; solve() only updates the parameters."""

    def __init__(self, v_max: float, slack_weight: float) -> None:
        self.u = cp.Variable(2)
        self.s = cp.Variable(nonneg=True)
        self.u_ref = cp.Parameter(2)
        self.n = cp.Parameter(2)
        self.b = cp.Parameter()
        cost = 0.5 * cp.sum_squares(self.u - self.u_ref) + slack_weight * self.s
        constraints = [self.n @ self.u <= self.b + self.s, cp.abs(self.u) <= v_max]
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, u_ref: np.ndarray, n: np.ndarray, b: float) -> tuple[np.ndarray | None, float]:
        self.u_ref.value = np.asarray(u_ref, dtype=float)
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
