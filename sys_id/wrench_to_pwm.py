#!/usr/bin/env python3
"""
PX4-style 6-DOF wrench allocator for BlueROV2 Heavy (8 thrusters).

Maps a body-frame wrench

    w = [tau_x, tau_y, tau_z, F_x, F_y, F_z]
         roll   pitch  yaw    surge sway  heave

to normalised per-thruster commands  u ∈ [-1, 1]^8  via:

    B  = 6×8 effectiveness matrix   (built from thruster geometry)
    u  = pinv(B) @ w                (pseudoinverse allocation)
    u  = clip(u, -1, 1)            (actuator saturation)

PX4 reference files
-------------------
Airframe geometry (CA_ROTOR* params):
    ROMFS/px4fmu_common/init.d-posix/airframes/1022_gazebo-classic_uuv_bluerov2_heavy
    ROMFS/px4fmu_common/init.d/airframes/60002_uuv_bluerov2_heavy

Effectiveness matrix construction:
    src/modules/control_allocator/VehicleActuatorEffectiveness/ActuatorEffectivenessRotors.cpp
        computeEffectivenessMatrix()  –  per-rotor column:
            axis_n  = axis / ||axis||
            thrust  = ct * axis_n
            moment  = ct * cross(position, axis_n) - ct * km * axis_n
        Rows 0-2 → moment (roll, pitch, yaw)
        Rows 3-5 → thrust (Fx, Fy, Fz)

Wrench vector ordering:
    src/modules/control_allocator/ControlAllocator.cpp  Run()
        c = [torque_x, torque_y, torque_z, thrust_x, thrust_y, thrust_z]

Pseudoinverse allocator:
    src/lib/control_allocation/control_allocation/ControlAllocationPseudoInverse.cpp
        allocate()  –  actuator_sp = trim + pinv(B) @ (control_sp - control_trim)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Thruster geometry — BlueROV2 Heavy
# ---------------------------------------------------------------------------
# Keep two BlueROV2 Heavy geometry variants:
# 1) "60002" (current gz/native BlueROV2 Heavy)
#    ROMFS/px4fmu_common/init.d-posix/airframes/60002_gz_uuv_bluerov2_heavy
#    ROMFS/px4fmu_common/init.d/airframes/60002_uuv_bluerov2_heavy
# 2) "1022" (older gazebo-classic BlueROV2 Heavy)
#    ROMFS/px4fmu_common/init.d-posix/airframes/1022_gazebo-classic_uuv_bluerov2_heavy
# Each tuple: (axis_x, axis_y, axis_z, pos_x, pos_y, pos_z, km)
#
# Rotors 0-3: horizontal thrusters (surge / sway / yaw)
# Rotors 4-7: vertical thrusters   (heave / roll / pitch)
#
# NOTE:
# - CA_R_REV = 255 (all 8 motors reversible) affects output limits and allows
#   negative actuator commands; it does not change effectiveness matrix terms.
# - CA_ROTOR{i}_CT is not set in the BlueROV2 Heavy airframe and therefore
#   falls back to module default (CA_ROTOR{i}_CT default = 6.5 in module.yaml).

_BLUEROV2_HEAVY_ROTORS_60002 = [
    #  AX   AY   AZ    PX    PY    PZ   KM
    (  1., -1.,  0.,   0.14,  0.10,  0.06, 0.),  # rotor 0
    (  1.,  1.,  0.,   0.14, -0.10,  0.06, 0.),  # rotor 1
    (  1.,  1.,  0.,  -0.14,  0.10,  0.06, 0.),  # rotor 2
    (  1., -1.,  0.,  -0.14, -0.10,  0.06, 0.),  # rotor 3
    (  0.,  0., -1.,   0.12,  0.22,  0.00, 0.),  # rotor 4
    (  0.,  0.,  1.,   0.12, -0.22,  0.00, 0.),  # rotor 5
    (  0.,  0.,  1.,  -0.12,  0.22,  0.00, 0.),  # rotor 6
    (  0.,  0., -1.,  -0.12, -0.22,  0.00, 0.),  # rotor 7
]

_BLUEROV2_HEAVY_ROTORS_1022 = [
    #  AX   AY   AZ    PX    PY    PZ   KM
    (  1., -1.,  0.,   0.5,  0.3,  0.2, 0.),  # rotor 0
    (  1.,  1.,  0.,   0.5, -0.3,  0.2, 0.),  # rotor 1
    (  1.,  1.,  0.,  -0.5,  0.3,  0.2, 0.),  # rotor 2
    (  1., -1.,  0.,  -0.5, -0.3,  0.2, 0.),  # rotor 3
    # 1022 sets only AZ for 4-6 and omits rotor7 AZ; resolve missing fields
    # using module.yaml defaults: AX=0, AY=0, AZ=-1, PZ=0.
    (  0.,  0., -1.,   0.5,  0.5,  0.0, 0.),  # rotor 4
    (  0.,  0.,  1.,   0.5, -0.5,  0.0, 0.),  # rotor 5
    (  0.,  0.,  1.,  -0.5,  0.5,  0.0, 0.),  # rotor 6
    (  0.,  0., -1.,  -0.5, -0.5,  0.0, 0.),  # rotor 7
]

_BLUEROV2_HEAVY_GEOMETRIES = {
    "60002": _BLUEROV2_HEAVY_ROTORS_60002,
    "1022": _BLUEROV2_HEAVY_ROTORS_1022,
}

# PX4 default CA_ROTOR{i}_CT from src/modules/control_allocator/module.yaml.
_DEFAULT_CT = 6.5


def build_bluerov2_heavy_effectiveness_matrix(
    ct: float | np.ndarray | None = None,
    geometry: str = "60002",
) -> np.ndarray:
    """Build the 6×8 effectiveness matrix B from BlueROV2 Heavy geometry.

    Mirrors ActuatorEffectivenessRotors::computeEffectivenessMatrix().

    Parameters
    ----------
    ct : float | np.ndarray | None
        Rotor thrust coefficient(s) used as in PX4:
          thrust = ct * axis
          moment = ct * cross(position, axis) - ct * km * axis
        If None, uses PX4 default CA_ROTOR{i}_CT = 6.5 for all 8 rotors.
        If float, applies same value to every rotor.
        If array-like, must provide one value per rotor.
    geometry : str
        Geometry variant key:
        - "60002": current BlueROV2 Heavy airframe values
        - "1022": gazebo-classic BlueROV2 Heavy values

    Returns
    -------
    B : np.ndarray, shape (6, 8)
        Rows 0-2 = torque effectiveness  [roll, pitch, yaw]
        Rows 3-5 = force  effectiveness  [Fx,   Fy,    Fz]
    """
    rotors = _BLUEROV2_HEAVY_GEOMETRIES.get(geometry)
    if rotors is None:
        raise ValueError(f"unknown geometry '{geometry}', choose one of {tuple(_BLUEROV2_HEAVY_GEOMETRIES)}")

    n_rotors = len(rotors)
    if ct is None:
        ct_per_rotor = np.full(n_rotors, _DEFAULT_CT)
    elif np.isscalar(ct):
        ct_per_rotor = np.full(n_rotors, float(ct))
    else:
        ct_per_rotor = np.asarray(ct, dtype=float)
        if ct_per_rotor.shape != (n_rotors,):
            raise ValueError(f"ct must have shape ({n_rotors},), got {ct_per_rotor.shape}")

    B = np.zeros((6, n_rotors))

    for i, (ax, ay, az, px, py, pz, km) in enumerate(rotors):
        ct_i = ct_per_rotor[i]
        axis = np.array([ax, ay, az])
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            continue
        axis = axis / norm

        position = np.array([px, py, pz])

        thrust = ct_i * axis
        moment = ct_i * np.cross(position, axis) - ct_i * km * axis

        B[0:3, i] = moment
        B[3:6, i] = thrust

    return B


def allocate_wrench(
    wrench: np.ndarray,
    B: np.ndarray | None = None,
) -> np.ndarray:
    """Allocate a 6-DOF wrench to 8 normalised thruster commands.

    Mirrors ControlAllocationPseudoInverse::allocate():
        u = pinv(B) @ w   then clip to [-1, 1].

    Parameters
    ----------
    wrench : array-like, shape (6,)
        [tau_x, tau_y, tau_z, F_x, F_y, F_z]
    B : np.ndarray, shape (6, 8), optional
        Effectiveness matrix.  Built from default geometry if *None*.
    geometry : str
        Passed to build_bluerov2_heavy_effectiveness_matrix() when B is None.

    Returns
    -------
    u : np.ndarray, shape (8,)
        Per-thruster commands clipped to [-1, 1].
    """



    w = np.asarray(wrench, dtype=float)
    # u = np.linalg.pinv(B @ A) @ w
    u = np.linalg.pinv(B) @ w
    return np.clip(u, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    test_cases = [
        ("Pure surge   (F_x = 1)", [0, 0, 0, 1, 0, 0]),
        ("Pure sway   (F_y = 1)", [0, 0, 0, 0, 1, 0]),
        ("Pure heave   (F_z = 1)", [0, 0, 0, 0, 0, 1]),
        ("Pure roll   (τ_x = 1)", [1, 0, 0, 0, 0, 0]),
        ("Pure pitch   (τ_y = 1)", [0, 1, 0, 0, 0, 0]),
        ("Pure yaw     (τ_z = 1)", [0, 0, 1, 0, 0, 0]),
    ]

    for geometry in ("60002", ):
        B = build_bluerov2_heavy_effectiveness_matrix(geometry=geometry)
        A = np.array([
            [0, 0, +1, 0, -1, +1],
            [0, 0, -1, 0, -1, -1],
            [0, 0, -1, 0, +1, +1],
            [0, 0, +1, 0, +1, -1],
            [+1, -1, 0, -1, 0, 0],
            [-1, -1, 0, -1, 0, 0],
            [+1, +1, 0, -1, 0, 0],
            [-1, +1, 0, -1, 0, 0],
        ])
        A_transpose = A.T
        print(A_transpose)
        A_new = np.concatenate(
            [A_transpose[:3, :], A_transpose[4:6, :], A_transpose[3:4, :]],
            axis=0,
        )
        print("matrix from ardupilot:")
        print(A_new)
        
        print("\nEffectiveness matrix B from px4:")
        # print("  rows 0-2 = torque [roll, pitch, yaw]")
        # print("  rows 3-5 = force  [Fx, Fy, Fz]")
        # print("  columns  = rotors 0 … 7\n")
        print(B)
        print()

        for label, w in test_cases:
            u = allocate_wrench(w, B)
            # 
            print(f"{label:30s}  →  u = {u}")
        print()


