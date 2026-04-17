import jax.numpy as jnp


def wrap_pi(a):
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


def _bluerov2_constants():
    # Physical/environment constants
    g = 9.82
    rho = 1000.0
    m = 13.5
    V = 0.0134
    Ix, Iy, Iz = 0.26, 0.23, 0.37
    xb = yb = 0.0
    zb = -0.01

    # Table A1 signed added-mass derivatives
    Xu_dot, Yv_dot, Zw_dot = -6.36, -7.12, -18.68
    Kp_dot, Mq_dot, Nr_dot = -0.189, -0.135, -0.222

    # Linear and quadratic damping magnitudes
    Xu, Yv, Zw = 13.7, 0.0, 33.0
    Kp, Mq, Nr = 0.0, 0.8, 0.0
    Xu_abs, Yv_abs, Zw_abs = 141.0, 217.0, 190.0
    Kp_abs, Mq_abs, Nr_abs = 1.19, 0.47, 1.5

    return {
        "g": g,
        "rho": rho,
        "m": m,
        "V": V,
        "Ix": Ix,
        "Iy": Iy,
        "Iz": Iz,
        "xb": xb,
        "yb": yb,
        "zb": zb,
        "Xu_dot": Xu_dot,
        "Yv_dot": Yv_dot,
        "Zw_dot": Zw_dot,
        "Kp_dot": Kp_dot,
        "Mq_dot": Mq_dot,
        "Nr_dot": Nr_dot,
        "Xu": Xu,
        "Yv": Yv,
        "Zw": Zw,
        "Kp": Kp,
        "Mq": Mq,
        "Nr": Nr,
        "Xu_abs": Xu_abs,
        "Yv_abs": Yv_abs,
        "Zw_abs": Zw_abs,
        "Kp_abs": Kp_abs,
        "Mq_abs": Mq_abs,
        "Nr_abs": Nr_abs,
    }


def _rotation_and_kinematics(eta):
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
    return R, T, J


def bluerov2_model_terms(eta, nu, v_current_ned=None):
    """Return the BlueROV2 model decomposition used by the controllers."""
    eta = jnp.asarray(eta, dtype=jnp.float32)
    nu = jnp.asarray(nu, dtype=jnp.float32)
    if v_current_ned is None:
        v_current_ned = jnp.zeros(3, dtype=eta.dtype)
    else:
        v_current_ned = jnp.asarray(v_current_ned, dtype=eta.dtype)

    c = _bluerov2_constants()
    R, T, J = _rotation_and_kinematics(eta)

    W = c["m"] * c["g"]
    B = c["rho"] * c["g"] * c["V"]

    MRB = jnp.diag(jnp.array([c["m"], c["m"], c["m"], c["Ix"], c["Iy"], c["Iz"]]))
    MA = -jnp.diag(
        jnp.array(
            [
                c["Xu_dot"],
                c["Yv_dot"],
                c["Zw_dot"],
                c["Kp_dot"],
                c["Mq_dot"],
                c["Nr_dot"],
            ]
        )
    )
    M = MRB + MA
    M_inv = jnp.linalg.inv(M)

    u, v, w, p, q, r = nu
    C_RB = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0, c["m"] * w, -c["m"] * v],
            [0.0, 0.0, 0.0, -c["m"] * w, 0.0, c["m"] * u],
            [0.0, 0.0, 0.0, c["m"] * v, -c["m"] * u, 0.0],
            [0.0, c["m"] * w, -c["m"] * v, 0.0, -c["Iz"] * r, -c["Iy"] * q],
            [-c["m"] * w, 0.0, c["m"] * u, c["Iz"] * r, 0.0, c["Ix"] * p],
            [c["m"] * v, -c["m"] * u, 0.0, c["Iy"] * q, -c["Ix"] * p, 0.0],
        ]
    )
    C_A = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0, -c["Zw_dot"] * w, c["Yv_dot"] * v],
            [0.0, 0.0, 0.0, c["Zw_dot"] * w, 0.0, -c["Xu_dot"] * u],
            [0.0, 0.0, 0.0, -c["Yv_dot"] * v, c["Xu_dot"] * u, 0.0],
            [0.0, -c["Zw_dot"] * w, c["Yv_dot"] * v, 0.0, -c["Nr_dot"] * r, c["Mq_dot"] * q],
            [c["Zw_dot"] * w, 0.0, -c["Xu_dot"] * u, c["Nr_dot"] * r, 0.0, -c["Kp_dot"] * p],
            [-c["Yv_dot"] * v, c["Xu_dot"] * u, 0.0, -c["Mq_dot"] * q, c["Kp_dot"] * p, 0.0],
        ]
    )
    C = C_RB + C_A

    v_c_body = R.T @ v_current_ned
    nu_rel = nu.at[:3].add(-v_c_body)
    u_rel, v_rel, w_rel, p_rel, q_rel, r_rel = nu_rel

    Dlin = jnp.diag(jnp.array([c["Xu"], c["Yv"], c["Zw"], c["Kp"], c["Mq"], c["Nr"]]))
    Dquad = jnp.diag(
        jnp.array(
            [
                c["Xu_abs"] * jnp.abs(u_rel),
                c["Yv_abs"] * jnp.abs(v_rel),
                c["Zw_abs"] * jnp.abs(w_rel),
                c["Kp_abs"] * jnp.abs(p_rel),
                c["Mq_abs"] * jnp.abs(q_rel),
                c["Nr_abs"] * jnp.abs(r_rel),
            ]
        )
    )
    D = Dlin + Dquad

    phi, theta = eta[3], eta[4]
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    cth, sth = jnp.cos(theta), jnp.sin(theta)
    g_vec = jnp.array(
        [
            (W - B) * sth,
            -(W - B) * cth * sphi,
            -(W - B) * cth * cphi,
            c["yb"] * B * cth * cphi - c["zb"] * B * cth * sphi,
            -c["zb"] * B * sth - c["xb"] * B * cth * cphi,
            c["xb"] * B * cth * sphi + c["yb"] * B * sth,
        ]
    )

    return {
        "eta": eta,
        "nu": nu,
        "R": R,
        "T": T,
        "J": J,
        "MRB": MRB,
        "MA": MA,
        "M": M,
        "M_inv": M_inv,
        "C_RB": C_RB,
        "C_A": C_A,
        "C": C,
        "D": D,
        "g": g_vec,
        "nu_rel": nu_rel,
    }


def bluerov2_dynamics(x, v_current_ned=None):
    """Affine dynamics ẋ = f(x) + G(x) τ for BlueROV2."""
    x = jnp.asarray(x, dtype=jnp.float32)
    eta = x[:6]
    nu = x[6:]

    terms = bluerov2_model_terms(eta, nu, v_current_ned=v_current_ned)
    eta_dot = terms["J"] @ nu
    nu_dot_drift = terms["M_inv"] @ (
        -terms["C"] @ nu - terms["D"] @ terms["nu_rel"] - terms["g"]
    )

    f = jnp.concatenate([eta_dot, nu_dot_drift])
    G_full = jnp.zeros((12, 6), dtype=x.dtype)
    G_full = G_full.at[6:, :].set(terms["M_inv"])
    return f, G_full
