from functools import partial

import jax
import jax.numpy as jnp

def wrap_pi(a):
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


def bluerov2_dynamics(x, v_current_ned=None):
    """Affine dynamics ẋ = f(x) + G(x) τ for BlueROV2.
    Returns:
      f: 12x1 drift dynamics at state x
      G: 12x6 input matrix mapping body wrench τ
    """
    # Physical/environment constants
    g = 9.82
    rho = 1000.0
    m = 13.5
    V = 0.0134
    Ix, Iy, Iz = 0.26, 0.23, 0.37
    xb = yb = 0.0
    zb = -0.01
    # Table A1 lists the signed added-mass derivatives used in Eq. (7).
    # They are negative for the BlueROV2, so -X_u_dot yields a positive
    # diagonal entry in M_A as expected from Fossen's convention.
    Xu_dot, Yv_dot, Zw_dot = -6.36, -7.12, -18.68
    Kp_dot, Mq_dot, Nr_dot = -0.189, -0.135, -0.222
    # Linear and quadratic damping magnitudes
    Xu, Yv, Zw = 13.7, 0.0, 33.0
    Kp, Mq, Nr = 0.0, 0.8, 0.0
    Xu_abs, Yv_abs, Zw_abs = 141.0, 217.0, 190.0
    Kp_abs, Mq_abs, Nr_abs = 1.19, 0.47, 1.5
    # Derived parameters
    W = m * g
    B = rho * g * V
    # Inertia matrices per Eqs. (6)–(7).
    MRB = jnp.diag(jnp.array([m, m, m, Ix, Iy, Iz]))
    MA = -jnp.diag(jnp.array([Xu_dot, Yv_dot, Zw_dot, Kp_dot, Mq_dot, Nr_dot]))
    M = MRB + MA
    M_inv = jnp.linalg.inv(M)

    eta = x[:6]
    nu = x[6:]

    phi, theta, psi = eta[3], eta[4], eta[5]
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    cth, sth = jnp.cos(theta), jnp.sin(theta)
    cpsi, spsi = jnp.cos(psi), jnp.sin(psi)
    R = jnp.array(
        [
            [
                cpsi * cth,
                -spsi * cphi + cpsi * sth * sphi,
                spsi * sphi + cpsi * cphi * sth,
            ],
            [
                spsi * cth,
                cpsi * cphi + sphi * sth * spsi,
                -cpsi * sphi + cphi * sth * spsi,
            ],
            [-sth, cth * sphi, cth * cphi],
        ]
    )

    eps = 1e-6
    sign = jnp.where(cth >= 0.0, 1.0, -1.0)
    cth_safe = jnp.where(jnp.abs(cth) < eps, sign * eps, cth)
    inv_cth = 1.0 / cth_safe
    tth = sth * inv_cth
    T = jnp.array(
        [
            [1.0, sphi * tth, cphi * tth],
            [0.0, cphi, -sphi],
            [0.0, sphi * inv_cth, cphi * inv_cth],
        ]
    )

    J = jnp.zeros((6, 6))
    J = J.at[:3, :3].set(R)
    J = J.at[3:, 3:].set(T)

    u, v, w, p, q, r = nu
    C_RB = jnp.array(
        [
            [0, 0, 0, 0, m * w, -m * v],
            [0, 0, 0, -m * w, 0, m * u],
            [0, 0, 0, m * v, -m * u, 0],
            [0, m * w, -m * v, 0, -Iz * r, -Iy * q],
            [-m * w, 0, m * u, Iz * r, 0, Ix * p],
            [m * v, -m * u, 0, Iy * q, -Ix * p, 0],
        ]
    )
    C_A = jnp.array(
        [
            [0, 0, 0, 0, -Zw_dot * w, Yv_dot * v],
            [0, 0, 0, Zw_dot * w, 0, -Xu_dot * u],
            [0, 0, 0, -Yv_dot * v, Xu_dot * u, 0],
            [0, -Zw_dot * w, Yv_dot * v, 0, -Nr_dot * r, Mq_dot * q],
            [Zw_dot * w, 0, -Xu_dot * u, Nr_dot * r, 0, -Kp_dot * p],
            [-Yv_dot * v, Xu_dot * u, 0, -Mq_dot * q, Kp_dot * p, 0],
        ]
    )
    C = C_RB + C_A

    # TODO: water current projected into body, Eq. (20)
    v_current_ned = jnp.zeros(3)
    v_c_body = R.T @ v_current_ned
    nu_rel = nu.at[:3].add(-v_c_body)
    u_rel, v_rel, w_rel, p_rel, q_rel, r_rel = nu_rel
    Dlin = jnp.diag(jnp.array([Xu, Yv, Zw, Kp, Mq, Nr]))
    Dquad = jnp.diag(
        jnp.array(
            [
                Xu_abs * jnp.abs(u_rel),
                Yv_abs * jnp.abs(v_rel),
                Zw_abs * jnp.abs(w_rel),
                Kp_abs * jnp.abs(p_rel),
                Mq_abs * jnp.abs(q_rel),
                Nr_abs * jnp.abs(r_rel),
            ]
        )
    )
    D = Dlin + Dquad

    g_vec = jnp.array(
        [
            (W - B) * sth,
            -(W - B) * cth * sphi,
            -(W - B) * cth * cphi,
            yb * B * cth * cphi - zb * B * cth * sphi,
            -zb * B * sth - xb * B * cth * cphi,
            xb * B * cth * sphi + yb * B * sth,
        ]
    )

    eta_dot = J @ nu
    # Drift term from Eq. (2): Mν̇ + Cν + Dν_r + g = τ  =>  ν̇ = M^{-1}( -Cν - Dν_r - g) + M^{-1} τ
    nu_dot_drift = M_inv @ (-C @ nu - D @ nu_rel - g_vec)
    f = jnp.concatenate([eta_dot, nu_dot_drift])
    G_full = jnp.zeros((12, 6))
    G_full = G_full.at[6:, :].set(M_inv)
    return f, G_full
