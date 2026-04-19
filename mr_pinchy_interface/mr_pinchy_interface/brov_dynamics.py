import numpy as np


def wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


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


def bluerov2_model_terms(eta, nu):
    """Return the BlueROV2 model decomposition used by the controllers."""
    eta = np.asarray(eta, dtype=np.float32)
    nu = np.asarray(nu, dtype=np.float32)

    c = _bluerov2_constants()

    W = c["m"] * c["g"]
    B = c["rho"] * c["g"] * c["V"]

    MRB = np.diag(np.array([c["m"], c["m"], c["m"], c["Ix"], c["Iy"], c["Iz"]]))
    MA = -np.diag(
        np.array(
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
    M_inv = np.linalg.inv(M)

    u, v, w, p, q, r = nu
    C_RB = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, c["m"] * w, -c["m"] * v],
            [0.0, 0.0, 0.0, -c["m"] * w, 0.0, c["m"] * u],
            [0.0, 0.0, 0.0, c["m"] * v, -c["m"] * u, 0.0],
            [0.0, c["m"] * w, -c["m"] * v, 0.0, -c["Iz"] * r, -c["Iy"] * q],
            [-c["m"] * w, 0.0, c["m"] * u, c["Iz"] * r, 0.0, c["Ix"] * p],
            [c["m"] * v, -c["m"] * u, 0.0, c["Iy"] * q, -c["Ix"] * p, 0.0],
        ]
    )
    C_A = np.array(
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

    nu_rel = nu
    u_rel, v_rel, w_rel, p_rel, q_rel, r_rel = nu_rel

    Dlin = np.diag(np.array([c["Xu"], c["Yv"], c["Zw"], c["Kp"], c["Mq"], c["Nr"]]))
    Dquad = np.diag(
        np.array(
            [
                c["Xu_abs"] * np.abs(u_rel),
                c["Yv_abs"] * np.abs(v_rel),
                c["Zw_abs"] * np.abs(w_rel),
                c["Kp_abs"] * np.abs(p_rel),
                c["Mq_abs"] * np.abs(q_rel),
                c["Nr_abs"] * np.abs(r_rel),
            ]
        )
    )
    D = Dlin + Dquad

    phi, theta = eta[3], eta[4]
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    g_vec = np.array(
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
