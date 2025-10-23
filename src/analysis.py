# Python code to compute and visualise all key results from the manuscript
# "Self-Instantiated Stress–Energy: A Predictive Framework for Matter and Metric in Closed-System Cosmology"
# experimental part. This script is self-contained, does not read/write files, and produces figures on screen.
# Charts: one plot per figure (no subplots), matplotlib only, no explicit colors.

import numpy as np

from src.modeling.physics_model import CosmologyParams, k_phys_to_code, compute_background_and_spectra, \
    conservation_residual, \
    stability_z2
from src.utils.file_system_utils import save_data
from src.utils.plot_utils import plot_charts

# -------------------------------------------
#     Run Matter Framework Modeling
# -------------------------------------------
if __name__ == "__main__":
    P = CosmologyParams(
        Omega_r0=9.2e-5, Omega_m0=0.315, Omega_L0=0.684,
        Omega_ent0=1e-5,  # smaller, so the background doesn't roll off too quickly into de Sitter
        epsilon=0.01,  # change w_ent more slowly
        DeltaN=50.0,  # long plateau => almost const ε_H
        N0=-3.0,  # act before pivot, so the pivot lies on a plateau
        c_s_scalar=1.0,
        n0=0.4, k0=k_phys_to_code(0.05), sigma_ln_k=0.4,
        Gamma_over_H=3.0, A_ring=0.02, phi_ring=0.0
    )
    P.finalize()

    # Original
    result = compute_background_and_spectra(
        P,
        N_min=-10.0, N_max=6.0, nN=4001,
        kmin=5e-4, kmax=1.0, nk=256
    )

    # Print a compact numeric summary
    resid = conservation_residual(result["N"], result["rho_ent"], result["w"])
    z2 = stability_z2(result["N"], result["a"], result["eps"], P.c_s_scalar)
    print("Stability: min z^2 =", float(np.min(z2)), "; c_s^2 =", P.c_s_scalar ** 2)
    # Draw all figures
    plot_charts(result)

    # Save analysis data to files
    save_data(result)
