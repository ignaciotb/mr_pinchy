"""Pure-python checks for the soft-min CBF and the CBF-QP (no ROS needed)."""
from __future__ import annotations

import math

import numpy as np

from mr_pinchy_interface.brov_safe_relative_core import CbfQp, SoftminCbf, rotation_2d

D_SAFE, ALPHA, KAPPA = 0.6, 1.0, 50.0


def _wall_world(x: float, n: int = 200) -> np.ndarray:
    """Static wall at x (world frame), spanning y in [-2, 2]."""
    return np.stack([np.full(n, x), np.linspace(-2.0, 2.0, n)], axis=1)


def _simulate(u_ref_fn, points_world, steps=600, dt=0.02, noise=0.01, seed=0):
    """Closed loop: world-frame truth, body-frame soft-min CBF + QP. Robot yaws at
    0.2 rad/s; the sonar (12.5 Hz, noisy) gives the points in the body frame.
    Returns (min_h_true, last_u)."""
    rng = np.random.default_rng(seed)
    cbf = SoftminCbf(D_SAFE, ALPHA, KAPPA)
    qp = CbfQp(v_max=0.3, slack_weight=1000.0)
    omega, yaw = 0.2, 0.0
    robot_w = np.zeros(2)
    min_h = math.inf
    u = np.zeros(2)
    for k in range(steps):
        R_wb = rotation_2d(yaw)  # body -> world
        if k % 4 == 0:
            points_b = (points_world - robot_w) @ R_wb + rng.normal(0.0, noise, points_world.shape)
        n, b, _ = cbf.constraint(points_b)
        u, _ = qp.solve(u_ref_fn(k), n, b)
        assert u is not None
        robot_w = robot_w + dt * (R_wb @ u)  # velocity command tracked perfectly
        yaw += omega * dt
        min_h = min(min_h, np.linalg.norm(points_world - robot_w, axis=1).min() - D_SAFE)
    return min_h, u


def test_softmin_is_a_tight_lower_bound_with_gradient_toward_nearest():
    pts = np.array([[1.0, 0.0], [2.0, 0.5], [-1.5, 1.0]])
    cbf = SoftminCbf(D_SAFE, ALPHA, KAPPA)
    n, b, h = cbf.constraint(pts)
    h_min = 1.0 - D_SAFE
    assert h_min - math.log(len(pts)) / KAPPA <= h <= h_min + 1e-5
    assert np.allclose(n, [1.0, 0.0], atol=0.05)  # unit vector toward the nearest point
    assert math.isclose(b, ALPHA * h)


def test_more_than_max_points_keeps_the_nearest():
    rng = np.random.default_rng(0)
    far = rng.uniform(2.0, 3.0, (5000, 2))
    near = np.array([[0.0, 1.0]])
    cbf = SoftminCbf(D_SAFE, ALPHA, KAPPA, max_points=256)
    n, _, h = cbf.constraint(np.vstack([far, near]))
    assert abs(h - (1.0 - D_SAFE)) < 0.05
    assert np.allclose(n, [0.0, 1.0], atol=0.05)


def test_far_from_obstacle_command_passes_through():
    cbf = SoftminCbf(D_SAFE, ALPHA, KAPPA)
    qp = CbfQp(v_max=0.3, slack_weight=1000.0)
    n, b, h = cbf.constraint(_wall_world(3.0))
    u, s = qp.solve(np.array([0.2, 0.1]), n, b)
    assert h > 2.0
    assert np.allclose(u, [0.2, 0.1], atol=1e-4)
    assert s < 1e-6


def test_head_on_wall_stays_safe():
    min_h, _ = _simulate(lambda k: np.array([0.3, 0.0]), _wall_world(2.0))
    assert min_h > -0.02


def test_driving_along_a_wall_is_not_blocked():
    # Wall on the left at y = 1.0, robot at 0.4 m from the safety distance, driving forward:
    # the soft-min gradient is the wall normal, so the forward command is untouched.
    cbf = SoftminCbf(D_SAFE, ALPHA, KAPPA)
    qp = CbfQp(v_max=0.3, slack_weight=1000.0)
    wall = np.stack([np.linspace(-2.0, 2.0, 200), np.full(200, 1.0)], axis=1)
    n, b, _ = cbf.constraint(wall)
    u, s = qp.solve(np.array([0.3, 0.0]), n, b)
    assert abs(n[0]) < 0.02 and n[1] > 0.9
    assert np.allclose(u, [0.3, 0.0], atol=1e-3)
    assert s < 1e-6


def test_smooth_weight_pulls_command_toward_previous():
    # No active constraint (wall far away): the minimizer of ½|u-u_ref|² + ½w|u-u_prev|²
    # is the weighted mean (u_ref + w u_prev) / (1 + w); w = 0 recovers u = u_ref.
    cbf = SoftminCbf(D_SAFE, ALPHA, KAPPA)
    n, b, _ = cbf.constraint(_wall_world(3.0))
    u_ref, u_prev = np.array([0.2, 0.0]), np.array([0.0, 0.2])
    u, _ = CbfQp(v_max=0.3, slack_weight=1000.0, smooth_weight=1.0).solve(u_ref, n, b, u_prev)
    assert np.allclose(u, [0.1, 0.1], atol=1e-4)
    u, _ = CbfQp(v_max=0.3, slack_weight=1000.0).solve(u_ref, n, b, u_prev)
    assert np.allclose(u, u_ref, atol=1e-4)
