# Python code to compute and visualise all key results from the manuscript
# "Self-Instantiated Stress–Energy: A Predictive Framework for Matter and Metric in Closed-System Cosmology"
# experimental part. This script is self-contained, does not read/write files, and produces figures on screen.
# Charts: one plot per figure (no subplots), matplotlib only, no explicit colors.

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


# Parameters
@dataclass
class CosmologyParams:
    """
    Class representing cosmological parameters including an entanglement fluid component.

    :ivar Omega_r0: Present-day density parameter for radiation.
    :type Omega_r0: float
    :ivar Omega_m0: Present-day density parameter for matter.
    :type Omega_m0: float
    :ivar Omega_L0: Present-day density parameter for dark energy.
    :type Omega_L0: float
    :ivar Omega_ent0: Present-day density parameter for the entanglement fluid.
    :type Omega_ent0: float
    :ivar Omega_k0: Present-day curvature density parameter (inferred if None).
    :type Omega_k0: float
    :ivar H0: Present-day Hubble constant [km/s/Mpc].
    :type H0: float
    :ivar epsilon: Equation-of-state deformation parameter for the entanglement fluid.
    :type epsilon: float
    :ivar DeltaN: Duration of entanglement-fluid influence in e-folds.
    :type DeltaN: float
    :ivar N0: Initial e-folding reference point for the entanglement-fluid epoch.
    :type N0: float
    :ivar c_s_ent: Dimensionless sound speed of the entanglement fluid (0 ≤ c_s_ent ≤ 1).
    :type c_s_ent: float
    :ivar c_s_scalar: Dimensionless sound speed for scalar perturbations.
    :type c_s_scalar: float
    :ivar n0: Amplitude of the decohered occupancy profile.
    :type n0: float
    :ivar k0: Central scale of the decohered log-normal bump.
    :type k0: float
    :ivar sigma_ln_k: Width of the log-normal profile in log(k) space.
    :type sigma_ln_k: float
    :ivar Gamma_over_H: Dimensionless ratio of decoherence damping rate to Hubble rate during ring-down.
    :type Gamma_over_H: float
    :ivar A_ring: Amplitude of the ring-down oscillations.
    :type A_ring: float
    :ivar phi_ring: Phase offset of the ring-down oscillations [radians].
    :type phi_ring: float
    """
    # Present-day density parameters (Ω_k0 inferred if None)
    Omega_r0: float = 9.2e-5
    Omega_m0: float = 0.315
    Omega_L0: float = 0.684
    Omega_ent0: float = 1.0e-3
    Omega_k0: float = None
    H0: float = 1.0  # H0 units

    # Entanglement-fluid parameters (w_ent and rho_ent)
    epsilon: float = 0.05
    DeltaN: float = 4.0
    N0: float = -5.0

    # Sound speeds
    c_s_ent: float = 1.0  # 0 ≤ c_s_ent^2 ≤ 1
    c_s_scalar: float = 1.0

    # Decohered occupancy profile \bar n_k: lognormal bump around k0
    n0: float = 0.5
    k0: float = 0.05
    sigma_ln_k: float = 0.4

    # Decoherence damping (Γ/H) for ring-down
    Gamma_over_H: float = 5.0

    # Ring-down amplitude (small, O(slow-roll)) and phase
    A_ring: float = 0.02
    phi_ring: float = 0.0

    def finalize(self):
        if self.Omega_k0 is None:
            self.Omega_k0 = 1.0 - (self.Omega_r0 + self.Omega_m0 + self.Omega_L0 + self.Omega_ent0)


C_kms = 299_792.458  # km/s


def k_phys_to_code(k_phys_Mpc_inv: float, H0_km_s_Mpc: float = 67.4) -> float:
    return k_phys_Mpc_inv / (H0_km_s_Mpc / C_kms)


# ---------------------------------------------------------------------------------------
# Background & helper functions. Formulas exactly as in the manuscript.
# ---------------------------------------------------------------------------------------
def w_ent(N: np.ndarray, epsilon: float, DeltaN: float, N0: float) -> np.ndarray:
    # w_ent(N) = -1 + ε exp(-(N - N0)/ΔN)
    return -1.0 + epsilon * np.exp(-(N - N0) / DeltaN)


def rho_ent_over_rhoc0(N: np.ndarray, N0: float, epsilon: float, DeltaN: float, Omega_ent0: float) -> np.ndarray:
    # ρ_ent(N) = ρ_ent(0) * exp{ -3 ε ΔN [ 1 - exp(-(N - N0)/ΔN) ] } ; return in units of ρ_c0: Ω_ent0 * ...
    return Omega_ent0 * np.exp(-3.0 * epsilon * DeltaN * (1.0 - np.exp(-(N - N0) / DeltaN)))


def H_over_H0(N: np.ndarray, P: CosmologyParams) -> np.ndarray:
    # H^2/H0^2 = Ω_r0 e^{-4N} + Ω_m0 e^{-3N} + Ω_Λ0 + Ω_ent0 exp[-3 ε ΔN (1 - e^{-(N - N0)/ΔN})] + Ω_k0 e^{-2N}
    term_r = P.Omega_r0 * np.exp(-4.0 * N)
    term_m = P.Omega_m0 * np.exp(-3.0 * N)
    term_L = P.Omega_L0 * np.ones_like(N)
    term_ent = rho_ent_over_rhoc0(N, P.N0, P.epsilon, P.DeltaN, P.Omega_ent0)
    term_k = P.Omega_k0 * np.exp(-2.0 * N)
    H2 = term_r + term_m + term_L + term_ent + term_k
    H2 = np.clip(H2, 1e-30, None)
    return np.sqrt(H2)


def eps_H(N: np.ndarray, HN: np.ndarray) -> np.ndarray:
    # ε_H = - d ln H / dN
    dN = N[1] - N[0];
    dlnH_dN = np.gradient(np.log(HN), dN, edge_order=2)
    return -dlnH_dN


def conformal_time(N: np.ndarray, HN: np.ndarray) -> np.ndarray:
    # dη/dN = 1/(aH), a = e^N ; integrate cumulatively
    a = np.exp(N)
    detadN = 1.0 / (a * HN)
    eta = np.zeros_like(N)
    for i in range(1, len(N)):
        eta[i] = eta[i - 1] + 0.5 * (detadN[i] + detadN[i - 1]) * (N[i] - N[i - 1])
    return eta


# ----------------------------------------------------------
# Primordial spectra (freeze-out and occupancy)
# ----------------------------------------------------------

def nbar_k(k: np.ndarray, n0: float, k0: float, sigma_ln_k: float) -> np.ndarray:
    # \bar n_k = n0 * exp( - (ln(k/k0))^2 / (2 σ^2) )
    return n0 * np.exp(- (np.log(k / k0) ** 2) / (2.0 * sigma_ln_k ** 2))


def find_N_star_for_k(k: float,
                      N: np.ndarray,
                      a: np.ndarray,
                      H: np.ndarray,
                      c_s_scalar: float,
                      tol: float = 1e-8,
                      max_iter: int = 64,
                      window_mask: Optional[np.ndarray] = None,
                      prefer: str = "rightmost",
                      N_guess: Optional[float] = None) -> Tuple[float, int]:
    """
    Solve c_s * k = a(N) * H(N) on a chosen window of N.
    - If `window_mask` is provided, roots are searched ONLY where window_mask is True.
    - If multiple brackets exist, choose 'rightmost' root (largest N) unless a continuous
      branch near `N_guess` is available, in which case choose the bracket containing
      the closest root to N_guess.
    Returns (N_star, j_star). If no root exists in the window, returns (np.nan, -1).
    """
    # restrict to window
    if window_mask is None:
        Nw, aw, Hw = N, a, H
        idx_map = np.arange(len(N))
    else:
        mask = np.asarray(window_mask, dtype=bool)
        if not np.any(mask):
            return float("nan"), -1
        Nw, aw, Hw = N[mask], a[mask], H[mask]
        idx_map = np.nonzero(mask)[0]

    f = c_s_scalar * k - aw * Hw
    s = np.sign(f)

    # Find all sign-change brackets in the window
    cand = np.where(s[:-1] * s[1:] <= 0)[0]
    if cand.size == 0:
        # No sign change: return NaN to let caller adjust k-band
        return float("nan"), -1

    # If N_guess is supplied, try to pick a bracket close to it first
    if N_guess is not None and np.isfinite(N_guess):
        # choose bracket whose [Nl,Nr] contains N_guess, else nearest by center
        centers = 0.5 * (Nw[cand] + Nw[cand + 1])
        i_best = int(np.argmin(np.abs(centers - N_guess)))
    else:
        # otherwise choose by preference
        i_best = int(cand[-1]) if prefer == "rightmost" else int(cand[0])

    # initial bracket
    i = i_best
    Nl, Nr = float(Nw[i]), float(Nw[i + 1])
    fl, fr = float(f[i]), float(f[i + 1])

    # If N_guess lies outside [Nl,Nr], try to re-center bracket to the closest one
    if N_guess is not None and np.isfinite(N_guess):
        # pick bracket that actually contains the root closest to N_guess
        # (search local neighborhood among all candidates)
        distances = []
        for j in cand:
            c = 0.5 * (Nw[j] + Nw[j + 1])
            distances.append(abs(c - N_guess))
        i = int(cand[int(np.argmin(distances))])
        Nl, Nr = float(Nw[i]), float(Nw[i + 1])
        fl, fr = float(f[i]), float(f[i + 1])

    # Robust bisection within the bracket
    for _ in range(max_iter):
        Nm = 0.5 * (Nl + Nr)
        fm = c_s_scalar * k - float(np.exp(Nm) * float(np.interp(Nm, Nw, Hw)))
        if abs(fm) < tol or (Nr - Nl) < tol:
            N_star = Nm
            break
        if np.sign(fl) * np.sign(fm) <= 0:
            Nr, fr = Nm, fm
        else:
            Nl, fl = Nm, fm
    else:
        N_star = 0.5 * (Nl + Nr)

    # map back to original index space for j_star
    j_star = -1
    if window_mask is None:
        j_star = max(0, min(len(N) - 2, int(np.searchsorted(N, N_star) - 1)))
    else:
        # position in the window then map to global
        j_local = max(0, min(len(Nw) - 2, int(np.searchsorted(Nw, N_star) - 1)))
        j_star = int(idx_map[j_local])

    return float(N_star), j_star


def Pzeta_at_k(k: float,
               N: np.ndarray,
               a: np.ndarray,
               H: np.ndarray,
               eta: np.ndarray,
               epsH: np.ndarray,
               P,  # your CosmologyParams object (uses c_s_scalar, A_ring, phi_ring, Gamma_over_H, etc.)
               H_scale: float = 1.0,
               nbar_func=None) -> dict:
    """
    Uses robust root-finding for N_*(k) and a global amplitude scale H_scale (from the pivot).
    """
    if nbar_func is None:
        nbar_func = lambda kk: nbar_k(kk, P.n0, P.k0, P.sigma_ln_k)

    # Rooted freeze-out
    N_star, j_star = find_N_star_for_k(k, N, a, H, P.c_s_scalar)
    # a_star = math.exp(N_star)
    H_star = H_scale * float(np.interp(N_star, N, H))
    eps_star = float(np.interp(N_star, N, epsH))
    c_s_star = P.c_s_scalar

    # Occupancy and scalar/tensor spectra
    nbar = float(nbar_func(np.array([k]))[0])
    amp = 1.0 + 2.0 * nbar
    Pz = amp * (H_star ** 2) / (8.0 * math.pi ** 2 * max(eps_star, 1e-16) * max(c_s_star, 1e-16))
    Pt = 2.0 * (H_star ** 2) / (math.pi ** 2)  # scales as H^2 -> consistent with r invariance under calibration
    r = Pt / max(Pz, 1e-300)
    n_t = -2.0 * eps_star
    ratio = r / (-8.0 * n_t) if n_t != 0 else float("nan")

    # Ring-down with proper k-dependence via η_* (now varies with k)
    eta0 = float(np.interp(P.N0, N, eta))
    eta_star = float(np.interp(N_star, N, eta))
    delta_eta = max(eta_star - eta0, 0.0)
    damp = math.exp(-P.Gamma_over_H * delta_eta * H_star)  # uses scaled H_*
    ring = 1.0 + P.A_ring * math.cos(2.0 * c_s_star * k * eta0 + P.phi_ring) * damp
    Pz_ring = Pz * ring

    return {
        "k": k, "N_star": N_star, "H_star": H_star, "eps_star": eps_star, "c_s_star": c_s_star,
        "nbar": nbar, "Pz": Pz, "Pz_ring": Pz_ring, "Pt": Pt, "r": r, "n_t": n_t,
        "consistency_ratio": ratio, "ring_damp": damp, "eta0": eta0, "eta_star": eta_star
    }



# Diagnostics
def stability_z2(N: np.ndarray, a: np.ndarray, epsH_arr: np.ndarray, c_s_scalar: float) -> np.ndarray:
    # z^2 = 2 a^2 ε_H / c_s^2 (M_pl=1)
    return 2.0 * (a ** 2) * epsH_arr / (c_s_scalar ** 2)


# Inflation window mask and k-band selector
def inflation_window_mask(N: np.ndarray, epsH_arr: np.ndarray, N0: float, eps_max: float = 1.0) -> np.ndarray:
    """Window selecting the post-act accelerating era: ε_H < 1 and N ≥ N0."""
    return (epsH_arr < eps_max) & (N >= N0)


def recommend_k_band_in_window(N: np.ndarray, a: np.ndarray, H: np.ndarray,
                               window_mask: np.ndarray, c_s_scalar: float = 1.0,
                               qlow: float = 0.10, qhigh: float = 0.90) -> Tuple[float, float]:
    """
    Choose k-range such that c_s k lies within the central quantiles of aH on the specified window.
    This guarantees a unique, physical freeze-out crossing on the desired branch.
    """
    aH_w = (a * H)[window_mask]
    lo = np.quantile(aH_w, qlow)
    hi = np.quantile(aH_w, qhigh)
    kmin = max(1e-16, 0.99 * lo / max(c_s_scalar, 1e-16))
    kmax = 1.01 * hi / max(c_s_scalar, 1e-16)
    return float(kmin), float(kmax)


def recommend_k_band(N: np.ndarray, a: np.ndarray, H: np.ndarray,
                     c_s_scalar: float = 1.0,
                     qlow: float = 0.05, qhigh: float = 0.95) -> Tuple[float, float]:
    """
    Returns (kmin, kmax) such that c_s * k lies strictly inside the central [qlow, qhigh]
    quantile band of aH. This ensures each k in [kmin, kmax] has a freeze-out crossing.
    """
    aH = a * H
    lo = np.quantile(aH, qlow)
    hi = np.quantile(aH, qhigh)
    # safety margins away from edges
    kmin = (0.99 * lo) / c_s_scalar
    kmax = (1.01 * hi) / c_s_scalar
    # guard against non-positive lo
    kmin = max(kmin, 1e-12)
    return float(kmin), float(kmax)


def calibrate_H_scale_at_pivot(k_piv: float,
                               N: np.ndarray, a: np.ndarray, H: np.ndarray, eta: np.ndarray, epsH: np.ndarray,
                               c_s_scalar: float, nbar_func,
                               A_s_target: float = 2.1e-9,
                               window_mask: 'Optional[np.ndarray]' = None) -> float:
    """
    Compute multiplicative scale 'S' such that, when H -> S*H, the scalar amplitude
    at k_piv satisfies P_zeta(k_piv) = A_s_target using
      P_ζ = (1+2 n̄_k) H_*^2 / (8 π^2 ε_* c_s_*).
    Returns S ≥ 0.  (Apply H_scaled = S * H everywhere; r stays invariant.)
    """
    # Find pivot freeze-out with root-finding:
    N_star, j = find_N_star_for_k(k_piv, N, a, H, c_s_scalar,
                                  window_mask=window_mask, prefer="rightmost")
    H_star_code = float(np.interp(N_star, N, H))
    eps_star = float(np.interp(N_star, N, epsH))
    nbar = float(nbar_func(np.array([k_piv]))[0])
    denom = 8.0 * np.pi ** 2 * max(eps_star, 1e-16) * max(c_s_scalar, 1e-16)
    # Target H_* from As:
    H_star_target = np.sqrt(max(A_s_target, 0.0) * denom / max(1.0 + 2.0 * nbar, 1e-16))
    S = H_star_target / max(H_star_code, 1e-30)
    return float(S)


# Utility to compute spectra from externally supplied N_star
def spectra_from_Nstar(k: float,
                       N_star: float,
                       N: np.ndarray, a: np.ndarray, H: np.ndarray, eta: np.ndarray, epsH: np.ndarray,
                       P: 'CosmologyParams', H_scale: float, nbar_func) -> dict:
    a_star = math.exp(N_star)
    H_star = H_scale * float(np.interp(N_star, N, H))
    eps_star = float(np.interp(N_star, N, epsH))
    c_s_star = P.c_s_scalar
    nbar = float(nbar_func(np.array([k]))[0])
    amp = 1.0 + 2.0 * nbar
    Pz = amp * (H_star ** 2) / (8.0 * math.pi ** 2 * max(eps_star, 1e-16) * max(c_s_star, 1e-16))
    Pt = 2.0 * (H_star ** 2) / (math.pi ** 2)
    r = Pt / max(Pz, 1e-300)
    n_t = -2.0 * eps_star
    ratio = r / (-8.0 * n_t) if n_t != 0 else float("nan")
    eta0 = float(np.interp(P.N0, N, eta))
    eta_star = float(np.interp(N_star, N, eta))
    delta_eta = max(eta_star - eta0, 0.0)
    damp = math.exp(-P.Gamma_over_H * delta_eta * H_star)
    ring = 1.0 + P.A_ring * math.cos(2.0 * c_s_star * k * eta0 + P.phi_ring) * damp
    Pz_ring = Pz * ring
    return {
        "k": k, "N_star": N_star, "H_star": H_star, "eps_star": eps_star, "c_s_star": c_s_star,
        "nbar": nbar, "Pz": Pz, "Pz_ring": Pz_ring, "Pt": Pt, "r": r, "n_t": n_t,
        "consistency_ratio": ratio, "ring_damp": damp, "eta0": eta0, "eta_star": eta_star
    }


# Main computation & plotting
def finite_diff_log_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute d ln y / d ln x with a 5-point stencil in the interior and
    3-point one-sided stencils at the ends. Assumes x is strictly monotone.
    """
    lx, ly = np.log(x), np.log(np.maximum(y, 1e-300))
    n = len(x)
    g = np.empty(n)
    # ends: 3-point
    g[0] = (ly[1] - ly[0]) / (lx[1] - lx[0])
    g[-1] = (ly[-1] - ly[-2]) / (lx[-1] - lx[-2])
    if n >= 5:
        for i in range(2, n - 2):
            # symmetric 5-point stencil for first derivative in ln space
            g[i] = (ly[i - 2] - 8 * ly[i - 1] + 8 * ly[i + 1] - ly[i + 2]) / (12 * (lx[i] - lx[i - 1]))
        # near-ends: 3-point centered
        g[1] = (ly[2] - ly[0]) / (lx[2] - lx[0])
        g[-2] = (ly[-1] - ly[-3]) / (lx[-1] - lx[-3])
    else:
        # fallback: simple gradient
        g[1:-1] = np.gradient(ly, lx)[1:-1]
    return g


def compute_background_and_spectra(P: CosmologyParams,
                                   N_min=-20.0,
                                   N_max=8.0,
                                   nN=3001,
                                   kmin=5e-3,
                                   kmax=0.5,
                                   nk=200) -> dict:
    P.finalize()
    N = np.linspace(N_min, N_max, nN)
    a = np.exp(N)
    H = H_over_H0(N, P) * P.H0
    eps = eps_H(N, H)
    w = w_ent(N, P.epsilon, P.DeltaN, P.N0)
    rho_ent = rho_ent_over_rhoc0(N, P.N0, P.epsilon, P.DeltaN, P.Omega_ent0)
    eta = conformal_time(N, H)
    # slow-roll derivative for filtering: η_H ≡ d ln ε / dN
    deps_dN = np.gradient(eps, N, edge_order=2)

    # acceleration window
    win = inflation_window_mask(N, eps, P.N0, eps_max=1.0)

    # Planck pivot in "code" units
    k_pivot_code = k_phys_to_code(0.05)  # 0.05 Mpc^{-1} → code units

    # k-band, guaranteeing a single intersection and including the pivot
    kmin_w, kmax_w = recommend_k_band_in_window(N, a, H, win, c_s_scalar=P.c_s_scalar,
                                                qlow=0.10, qhigh=0.90)
    if not (kmin_w < k_pivot_code < kmax_w):
        # carefully expand the quantile band, but without going beyond the win window
        for qlow, qhigh in [(0.05, 0.95), (0.02, 0.98), (0.01, 0.99)]:
            kmin_w, kmax_w = recommend_k_band_in_window(N, a, H, win, P.c_s_scalar, qlow, qhigh)
            if kmin_w < k_pivot_code < kmax_w:
                break
        # if still outside, minimally "pull" the boundaries to the pivot
        kmin_w = min(kmin_w, 0.7 * k_pivot_code)
        kmax_w = max(kmax_w, 1.3 * k_pivot_code)

    kmin, kmax = kmin_w, kmax_w
    # Physically admissible window: require |η_H| = |d ln ε / dN| to be modest at freeze-out (no fit — applicability condition)
    etaH_cut = 0.05

    # calibration of the overall amplitude by A_s on the same pivot and in the same window
    H_scale = calibrate_H_scale_at_pivot(
        k_pivot_code, N, a, H, eta, eps, P.c_s_scalar,
        nbar_func=lambda kk: nbar_k(kk, P.n0, P.k0, P.sigma_ln_k),
        A_s_target=2.1e-9, window_mask=win)
    ks = np.geomspace(kmin, kmax, nk)

    rows = []
    N_prev = None
    for k in ks:
        N_star, j_star = find_N_star_for_k(float(k), N, a, H, P.c_s_scalar,
                                           window_mask=win, prefer="rightmost", N_guess=N_prev)
        if not np.isfinite(N_star):
            continue
        # local slow-roll parameters at freeze-out
        eps_loc = float(np.interp(N_star, N, eps))
        if eps_loc <= 0.0:
            continue
        deps_loc = float(np.interp(N_star, N, deps_dN))
        etaH_loc = deps_loc / max(eps_loc, 1e-30)  # η_H ≡ d ln ε / dN
        # Filter out modes that freeze out outside the applicability regime (|η_H| too large)
        # if abs(etaH_loc) > etaH_cut:
        #     continue
        row = spectra_from_Nstar(float(k), N_star, N, a, H, eta, eps, P, H_scale,
                                 nbar_func=lambda kk: nbar_k(kk, P.n0, P.k0, P.sigma_ln_k))
        rows.append(row)
        N_prev = N_star

    assert max(r["H_star"] for r in rows) < 1e-2, "Semi-classical bound violated: H_*/M_pl too large"

    Pz_ring = np.array([r["Pz_ring"] for r in rows])
    Pz = np.array([r["Pz"] for r in rows])
    Pt = np.array([r["Pt"] for r in rows])
    rvals = np.array([r["r"] for r in rows])
    n_t = np.array([r["n_t"] for r in rows])
    ratio = np.array([r["consistency_ratio"] for r in rows])
    nbar = np.array([r["nbar"] for r in rows])
    ring_damp = np.array([r["ring_damp"] for r in rows])
    N_star = np.array([r["N_star"] for r in rows])

    # Tilt estimate: n_s - 1 = d ln P / d ln k (use ring-corrected)
    lnk = np.log(ks[:len(Pz_ring)])
    lnP = np.log(Pz_ring)
    n_s_minus_1 = finite_diff_log_slope(ks[:len(Pz_ring)], np.array([r["Pz_ring"] for r in rows]))

    # Optional: provide η_H at each accepted N_* for diagnostics
    etaH_vals = np.array([float(np.interp(r["N_star"], N, deps_dN)) / max(float(np.interp(r["N_star"], N, eps)), 1e-30)
                          for r in rows])

    return dict(P=P, N=N, a=a, H=H, eps=eps, w=w, rho_ent=rho_ent, eta=eta,
                ks=ks[:len(Pz_ring)], Pz=Pz, Pz_ring=Pz_ring, Pt=Pt, r=rvals, n_t=n_t,
                ratio=ratio, nbar=nbar, ring_damp=ring_damp, N_star=N_star,
                n_s_minus_1=n_s_minus_1,
                etaH_vals=etaH_vals)


def conservation_residual(N: np.ndarray, rho_ent: np.ndarray, w_ent_arr: np.ndarray) -> np.ndarray:
    # Residual of: d ln ρ_ent / dN + 3(1+w_ent) = 0
    # Guard against log(0) to avoid NaN in diagnostics:
    dlnrho_dN = np.gradient(np.log(np.clip(rho_ent, 1e-300, None)), N, edge_order=2)
    R = dlnrho_dN + 3.0 * (1.0 + w_ent_arr)
    R[0] = np.nan
    R[-1:] = np.nan

    return R
