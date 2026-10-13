"""Pure-python checks for the relative-state KF and the CBF-QP (no ROS needed)."""
from __future__ import annotations

import math

import numpy as np

from mr_pinchy_interface.brov_safe_relative_core import (
    CbfQp,
    RelativeKF,
    cbf_constraint,
    nearest_cluster_centroid,
    rotation_2d,
)


def _simulate(u_ref_fn, p0, v_obs_world, d_safe, alpha, steps=600, dt=0.02, noise=0.03, seed=0):
    """Closed loop: world-frame truth, body-frame KF + QP. Robot yaws at 0.2 rad/s.
    Returns (min_h_true, kf_pos_err, kf_vel_err, last_u)."""
    rng = np.random.default_rng(seed)
    kf = RelativeKF(q_pos=0.01, q_vel=0.1, r_meas=noise)
    qp = CbfQp(v_max=0.3, slack_weight=1000.0)
    omega = 0.2
    yaw = 0.0
    robot_w = np.zeros(2)
    obs_w = np.array(p0, dtype=float)
    v_obs_world = np.asarray(v_obs_world, dtype=float)
    min_h = math.inf
    u = np.zeros(2)
    for k in range(steps):
        R_wb = rotation_2d(yaw)  # body -> world
        p_true_b = R_wb.T @ (obs_w - robot_w)
        if k == 0:
            kf.reset(p_true_b)
        elif k % 4 == 0:  # sonar at 12.5 Hz
            kf.update(p_true_b + rng.normal(0.0, noise, 2))
        n, b, _ = cbf_constraint(kf.p, kf.v_obs, d_safe, alpha)
        u, _ = qp.solve(u_ref_fn(k), n, b)
        assert u is not None
        # advance truth (robot velocity command tracked perfectly)
        robot_w = robot_w + dt * (R_wb @ u)
        obs_w = obs_w + dt * v_obs_world
        yaw += omega * dt
        kf.predict(dt, u, omega)
        min_h = min(min_h, np.linalg.norm(obs_w - robot_w) - d_safe)
    R_wb = rotation_2d(yaw)
    pos_err = np.linalg.norm(kf.p - R_wb.T @ (obs_w - robot_w))
    vel_err = np.linalg.norm(kf.v_obs - R_wb.T @ v_obs_world)
    return min_h, pos_err, vel_err, u


def test_kf_tracks_moving_obstacle_while_robot_turns():
    _, pos_err, vel_err, _ = _simulate(
        lambda k: np.array([0.0, 0.0]), p0=[2.5, 0.5], v_obs_world=[0.1, -0.05], d_safe=0.9, alpha=1.0
    )
    assert pos_err < 0.05
    assert vel_err < 0.05


def test_far_from_obstacle_command_passes_through():
    kf = RelativeKF(0.01, 0.1, 0.1)
    kf.reset(np.array([3.0, 0.0]))
    qp = CbfQp(v_max=0.3, slack_weight=1000.0)
    n, b, h = cbf_constraint(kf.p, kf.v_obs, d_safe=0.9, alpha=1.0)
    u, s = qp.solve(np.array([0.2, 0.1]), n, b)
    assert h > 2.0
    assert np.allclose(u, [0.2, 0.1], atol=1e-4)
    assert s < 1e-6


def test_head_on_static_obstacle_stays_safe():
    min_h, _, _, _ = _simulate(
        lambda k: np.array([0.2, 0.0]), p0=[2.0, 0.0], v_obs_world=[0.0, 0.0], d_safe=0.9, alpha=1.0
    )
    assert min_h > -0.02


def test_approaching_obstacle_stays_safe():
    min_h, _, _, _ = _simulate(
        lambda k: np.array([0.2, 0.0]), p0=[2.5, 0.0], v_obs_world=[-0.15, 0.0], d_safe=0.9, alpha=1.0
    )
    assert min_h > -0.05


def test_nearest_cluster_is_chosen():
    rng = np.random.default_rng(1)
    near = rng.normal([1.0, 0.0], 0.02, (30, 2))
    far = rng.normal([2.5, 0.5], 0.02, (80, 2))
    noise = rng.uniform(-3, 3, (5, 2))
    c = nearest_cluster_centroid(np.vstack([far, near, noise]), eps=0.1, min_pts=10)
    assert c is not None
    assert np.linalg.norm(c - [1.0, 0.0]) < 0.05
    assert nearest_cluster_centroid(noise, eps=0.1, min_pts=10) is None
