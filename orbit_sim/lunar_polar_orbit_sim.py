import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import argparse
import os
from datetime import datetime

# ---- SPICE / spiceypy ----
try:
    import spiceypy as sp
except ImportError:
    sp = None


# ============================================================
# Polar orbit rendezvous visualization + TXT export + SPICE
#
# - TXT export: Moon-centered (for your ray tracer)
# - Plot: Sun-centered J2000 frame using SPICE
#   * Sun at origin
#   * Moon at its J2000 position on Jan 25, 2027 (default)
#   * Target + chaser as small cluster around the Moon
# ============================================================

# ---- Physical constants ----
MU_MOON   = 4902.800066      # [km^3/s^2] lunar GM
R_MOON    = 1737.4           # [km] mean lunar radius
R_SUN     = 696_340.0        # [km] approximate solar radius

# ---- Orbit definition (will be overridden by CLI) ----
PERI_ALT  = 3000.0           # [km] periapsis altitude above surface
APO_ALT   = 20000.0          # [km] apoapsis altitude above surface

rp = R_MOON + PERI_ALT       # [km] periapsis radius
ra = R_MOON + APO_ALT        # [km] apoapsis radius
a  = 0.5 * (rp + ra)         # [km] semi-major axis
e  = 1.0 - rp / a            # eccentricity

# Polar orbit
i_deg = 90.0
i_rad = np.deg2rad(i_deg)

# RAAN = 0
Omega_deg = 0.0
Omega_rad = np.deg2rad(Omega_deg)

# Periapsis over SOUTH pole (target fixed here)
omega_deg = 90.0
omega_rad = np.deg2rad(omega_deg)


# ------------------------------------------------
# Kepler helpers
# ------------------------------------------------
def solve_kepler_E(M, e, tol=1e-10, max_iter=50):
    """
    Solve Kepler's equation M = E - e sin E for E (elliptic orbit).
    M can be scalar or numpy array.
    """
    M = np.array(M, dtype=float)
    E = M.copy()  # initial guess

    for _ in range(max_iter):
        f   = E - e * np.sin(E) - M
        fp  = 1.0 - e * np.cos(E)
        dE  = -f / fp
        E  += dE
        if np.all(np.abs(dE) < tol):
            break

    return E


def true_anomaly_from_E(E, e):
    """Convert eccentric anomaly E -> true anomaly f."""
    cosE = np.cos(E)
    sinE = np.sin(E)
    sqrt_1_me2 = np.sqrt(1.0 - e**2)

    cosf = (cosE - e) / (1.0 - e * cosE)
    sinf = (sqrt_1_me2 * sinE) / (1.0 - e * cosE)

    f = np.arctan2(sinf, cosf)
    return f


def coe_to_rv(a, e, i, Omega, omega, f, mu=MU_MOON):
    """
    Classical orbital elements -> position, velocity in inertial frame.
    Returns r, v as 3-element numpy arrays (km, km/s).
    """
    # Distance
    r_mag = a * (1.0 - e**2) / (1.0 + e * np.cos(f))

    # Perifocal position & velocity
    r_pf = np.array([
        r_mag * np.cos(f),
        r_mag * np.sin(f),
        0.0
    ])

    h = np.sqrt(mu * a * (1.0 - e**2))
    v_pf = (mu / h) * np.array([
        -np.sin(f),
        e + np.cos(f),
        0.0
    ])

    # Rotation matrix from perifocal to inertial
    cO, sO = np.cos(Omega), np.sin(Omega)
    co, so = np.cos(omega), np.sin(omega)
    ci, si = np.cos(i), np.sin(i)

    R3_Omega = np.array([
        [ cO, -sO, 0.0],
        [ sO,  cO, 0.0],
        [0.0, 0.0, 1.0]
    ])

    R1_i = np.array([
        [1.0, 0.0, 0.0],
        [0.0,  ci,  si],
        [0.0, -si,  ci]
    ])

    R3_omega = np.array([
        [ co, -so, 0.0],
        [ so,  co, 0.0],
        [0.0, 0.0, 1.0]
    ])

    Q = R3_Omega @ R1_i @ R3_omega  # perifocal -> inertial

    r_ijk = Q @ r_pf
    v_ijk = Q @ v_pf

    return r_ijk, v_ijk


# ------------------------------------------------
# Trajectory generator (Moon-centered)
# ------------------------------------------------
def generate_polar_rendezvous_trajectory(
    t_start=-600.0,    # [s]
    t_end=0.0,         # [s]
    dt=1.0             # [s]
):
    """
    Generate (Moon-centered):
        t_arr         : [N] time array from t_start to t_end in steps of dt
        r_target_arr  : [N, 3] km (frozen at periapsis)
        r_chaser_arr  : [N, 3] km (moving toward rendezvous)
    """
    n = np.sqrt(MU_MOON / a**3)

    N = int(round((t_end - t_start) / dt)) + 1
    t_arr = t_start + dt * np.arange(N)

    f_peri = 0.0
    r_peri, _ = coe_to_rv(a, e, i_rad, Omega_rad, omega_rad, f_peri)
    r_target_arr = np.repeat(r_peri[None, :], N, axis=0)

    r_chaser_arr = np.zeros((N, 3))

    for k, t in enumerate(t_arr):
        M = n * t
        E = solve_kepler_E(M, e)
        f = true_anomaly_from_E(E, e)
        r_ijk, _ = coe_to_rv(a, e, i_rad, Omega_rad, omega_rad, f)
        r_chaser_arr[k, :] = r_ijk

    return t_arr, r_target_arr, r_chaser_arr


# ------------------------------------------------
# TXT export (Moon-centered!)
# ------------------------------------------------
def export_txt(t_arr, r_target, r_chaser, filename):
    """
    Export camera/model states in Moon-centered frame
    (this is what your ray tracer expects).
    """
    with open(filename, "w") as f:
        f.write("# cam_x  cam_y  cam_z     model_x model_y model_z     yaw  pitch  roll\n")

        for cam, mdl in zip(r_chaser, r_target):
            cam_x, cam_y, cam_z = cam
            mdl_x, mdl_y, mdl_z = mdl

            yaw   = 0.0
            pitch = 0.0
            roll  = 0.0

            line = (
                f"{cam_x: .6f}  {cam_y: .6f}  {cam_z: .6f}    "
                f"{mdl_x: .6f}  {mdl_y: .6f}  {mdl_z: .6f}    "
                f"{yaw: .1f}  {pitch: .1f}  {roll: .1f}\n"
            )
            f.write(line)

    print(f"[+] Wrote {filename}")


# ------------------------------------------------
# SPICE helpers
# ------------------------------------------------
def get_moon_pos_from_sun(et, frame="J2000"):
    """
    Get Moon position relative to Sun in given frame using SPICE.

    Returns:
        r_moon_sun : 3-vector [km], Sun-centered.
    """
    if sp is None:
        raise RuntimeError(
            "spiceypy is not installed. Install with `pip install spiceypy`."
        )

    # State of MOON relative to SUN, in 'frame'
    state, lt = sp.spkezr("MOON", et, frame, "NONE", "SUN")
    r_moon_sun = np.array(state[0:3])  # [km]
    return r_moon_sun


# ------------------------------------------------
# Plot / animation helpers
# ------------------------------------------------
def set_equal_3d(ax, X, Y, Z, margin=0.1):
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)
    z_min, z_max = np.min(Z), np.max(Z)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if max_range == 0:
        max_range = 1.0

    x_center = 0.5 * (x_max + x_min)
    y_center = 0.5 * (y_max + y_min)
    z_center = 0.5 * (z_max + z_min)

    half = 0.5 * max_range * (1.0 + margin)

    ax.set_xlim(x_center - half, x_center + half)
    ax.set_ylim(y_center - half, y_center + half)
    ax.set_zlim(z_center - half, z_center + half)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Lunar polar rendezvous viz + TXT export\n"
            "Moon-centered TXT, Sun-centered plot via SPICE."
        )
    )
    parser.add_argument(
        "--time",
        type=float,
        default=10.0,
        help="How far back from docking to start [s] (positive, t_start = -time).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time step [s] for TXT export.",
    )
    parser.add_argument(
        "--peri_alt",
        type=float,
        default=15.0,
        help="Periapsis altitude above lunar surface [km].",
    )
    parser.add_argument(
        "--apo_alt",
        type=float,
        default=100.0,
        help="Apoapsis altitude above lunar surface [km].",
    )
    parser.add_argument(
        "--incl",
        type=float,
        default=90.0,
        help="Inclination [deg].",
    )
    parser.add_argument(
        "--raan",
        type=float,
        default=0.0,
        help="RAAN [deg].",
    )
    parser.add_argument(
        "--argp",
        type=float,
        default=90.0,
        help="Argument of periapsis [deg].",
    )
    parser.add_argument(
        "--txt",
        type=str,
        default=None,
        help="Output TXT filename (default auto: rendezvous_<time>s_dt<dt>s.txt).",
    )
    parser.add_argument(
        "--meta",
        type=str,
        required=True,
        help="SPICE meta-kernel (.tm) or any file in the folder containing naif0012.tls and de440s.bsp.",
    )
    parser.add_argument(
        "--epoch_utc",
        type=str,
        default="2027-01-25T00:00:00",
        help="UTC epoch for SPICE (default: 2027-01-25T00:00:00).",
    )

    args = parser.parse_args()

    # --- Check SPICE availability ---
    if sp is None:
        raise SystemExit(
            "ERROR: spiceypy is not available. Install with `pip install spiceypy`."
        )

    # --- Load SPICE kernels (Plan B: ignore meta contents, use its folder) ---
    kernel_dir = os.path.dirname(os.path.abspath(args.meta))
    sp.furnsh(os.path.join(kernel_dir, 'naif0012.tls'))
    sp.furnsh(os.path.join(kernel_dir, 'de440s.bsp'))

    # --- Convert epoch to ET (seconds past J2000) ---
    et0 = sp.utc2et(args.epoch_utc)

    # --- Get Moon's position relative to Sun at that epoch ---
    r_moon_sun = get_moon_pos_from_sun(et0, frame="J2000")  # [km]

    # Override orbit globals from CLI
    global PERI_ALT, APO_ALT, rp, ra, a, e
    global i_deg, i_rad, Omega_deg, Omega_rad, omega_deg, omega_rad

    PERI_ALT = args.peri_alt
    APO_ALT  = args.apo_alt
    rp = R_MOON + PERI_ALT
    ra = R_MOON + APO_ALT
    a  = 0.5 * (rp + ra)
    e  = 1.0 - rp / a

    i_deg = args.incl
    i_rad = np.deg2rad(i_deg)

    Omega_deg = args.raan
    Omega_rad = np.deg2rad(Omega_deg)

    omega_deg = args.argp
    omega_rad = np.deg2rad(omega_deg)

    t_start = -abs(args.time)
    t_end   = 0.0
    dt      = args.dt

    if args.txt is None:
        txt_name = f"rendezvous_{int(abs(args.time))}s_dt{int(dt)}s.txt"
    else:
        txt_name = args.txt

    # --- Generate Moon-centered trajectory ---
    t_arr, r_target_mc, r_chaser_mc = generate_polar_rendezvous_trajectory(
        t_start=t_start,
        t_end=t_end,
        dt=dt
    )

    # ---- Transform Moon-centered -> Sun-centered for TXT ----
    # r_moon_sun is a 3-vector [km] from SPICE (Sun-centered J2000)
    r_moon_sun_vec_full = r_moon_sun.reshape(1, 3)  # shape (1,3) for broadcasting
    r_target_sc = r_target_mc + r_moon_sun_vec_full   # target in Sun frame
    r_chaser_sc = r_chaser_mc + r_moon_sun_vec_full   # chaser in Sun frame

    # TXT now uses Sun-centered coordinates
    export_txt(t_arr, r_target_sc, r_chaser_sc, txt_name)
    print("[+] TXT written in Sun-centered frame (Sun at origin)")

    # --- Visualization trajectory (coarser dt for speed) ---
    dt_viz = max(dt, 10.0)
    t_arr_viz, r_target_viz_mc, r_chaser_viz_mc = generate_polar_rendezvous_trajectory(
        t_start=t_start,
        t_end=t_end,
        dt=dt_viz
    )

    # ---- Transform Moon-centered -> Sun-centered J2000 ----
    # We assume the Moon-centered inertial frame we used for the orbit
    # is aligned with J2000 at this epoch (just a visualization assumption).
    # So we just translate by r_moon_sun.
    r_moon_sun_vec = r_moon_sun.reshape(1, 3)

    cam_sun = r_chaser_viz_mc + r_moon_sun_vec   # chaser in Sun frame
    mdl_sun = r_target_viz_mc + r_moon_sun_vec   # target in Sun frame

    # For plotting convenience:
    cam = cam_sun
    mdl = mdl_sun

    cam_x, cam_y, cam_z = cam[:, 0], cam[:, 1], cam[:, 2]
    mdl_x, mdl_y, mdl_z = mdl[:, 0], mdl[:, 1], mdl[:, 2]

    # --- Build figure ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(
        "Sun-centered J2000: Moon, Target, Chaser\n"
        f"Epoch: {args.epoch_utc}"
    )

    # ---- Sun sphere at origin ----
    u_sun = np.linspace(0, 2*np.pi, 32)
    v_sun = np.linspace(0, np.pi, 16)
    sun_x = R_SUN * np.outer(np.cos(u_sun), np.sin(v_sun))
    sun_y = R_SUN * np.outer(np.sin(u_sun), np.sin(v_sun))
    sun_z = R_SUN * np.outer(np.ones_like(u_sun), np.cos(v_sun))
    ax.plot_surface(sun_x, sun_y, sun_z, alpha=0.3, linewidth=0)

    # ---- Moon sphere at r_moon_sun ----
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    moon_x = R_MOON * np.outer(np.cos(u), np.sin(v)) + r_moon_sun[0]
    moon_y = R_MOON * np.outer(np.sin(u), np.sin(v)) + r_moon_sun[1]
    moon_z = R_MOON * np.outer(np.ones_like(u), np.cos(v)) + r_moon_sun[2]
    ax.plot_surface(moon_x, moon_y, moon_z, alpha=0.4, linewidth=0)

    # ---- Orbit path (chaser) around the Moon in Sun frame ----
    ax.plot(cam_x, cam_y, cam_z, linestyle='--', linewidth=1)

    # Target marker (fixed at periapsis in this sim)
    target_marker, = ax.plot(
        [mdl_x[0]], [mdl_y[0]], [mdl_z[0]],
        marker='o', markersize=7
    )

    # Chaser markers
    chaser_marker, = ax.plot(
        [cam_x[0]], [cam_y[0]], [cam_z[0]],
        marker='o', markersize=7
    )
    chaser_traj, = ax.plot([], [], [], linewidth=2)

    ax.set_xlabel("X [km] (Sun-centered J2000)")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")

    # Include Sun + Moon + trajectories for scaling
    all_x = np.concatenate([cam_x, mdl_x, moon_x.flatten(), sun_x.flatten()])
    all_y = np.concatenate([cam_y, mdl_y, moon_y.flatten(), sun_y.flatten()])
    all_z = np.concatenate([cam_z, mdl_z, moon_z.flatten(), sun_z.flatten()])
    set_equal_3d(ax, all_x, all_y, all_z, margin=0.2)

    # Default view
    ax.view_init(elev=20, azim=40)

    # ---- Animation update ----
    def update(frame):
        k = frame
        chaser_marker.set_data([cam_x[k]], [cam_y[k]])
        chaser_marker.set_3d_properties([cam_z[k]])

        chaser_traj.set_data(cam_x[:k+1], cam_y[:k+1])
        chaser_traj.set_3d_properties(cam_z[:k+1])

        return chaser_marker, chaser_traj

    frames = len(t_arr_viz)
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

    plt.show()

    # Clean up SPICE
    sp.kclear()


if __name__ == "__main__":
    main()
