import json
import os

import numpy as np

from src.modeling.physics_model import conservation_residual, stability_z2, k_phys_to_code


def _write_csv(path: str, columns_dict: dict[str, np.ndarray]) -> None:
    """
    Writes data encapsulated in a dictionary of columns to a CSV file. Each key in the
    dictionary represents a column name, and its associated value is an array of data
    values. The function ensures that all columns have the same length before compiling
    them into a CSV format.

    :param path: The file path where the CSV should be written.
    :type path: str
    :param columns_dict: A dictionary where keys are column names as strings, and
        values are NumPy arrays holding the corresponding column data.
    :type columns_dict: dict[str, np.ndarray]
    :return: None
    """
    keys = list(columns_dict.keys())
    arrays = [np.asarray(columns_dict[k]).ravel() for k in keys]
    n = len(arrays[0])
    assert all(len(a) == n for a in arrays), "CSV columns must have equal length"
    data = np.column_stack(arrays)
    header = ",".join(keys)
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.10e")


def save_data(res: dict, out_dir_rel: str = "../experimental_data/data") -> list[str]:
    """
    Saves data derived from scientific calculations into multiple CSV files and JSON/NPZ bundles for analysis
    and figure generation. It also creates a manifest file mapping CSV filenames to the corresponding output
    figures.

    :param res: Dictionary containing computed results and data arrays. Includes background evolution variables
        like "N", "a", "H", etc., and k-dependent results such as "ks", "Pz", "r", etc.
    :type res: dict

    :param out_dir_rel: Relative path to the directory for saving output files. Default is
        "../experimental_data/data".
    :type out_dir_rel: str, optional

    :return: A list of strings indicating the lines of a generated manifest file that maps
        the output CSV files to their corresponding figures.
    :rtype: list[str]
    """
    P = res["P"]
    current_dir = os.path.dirname(__file__)
    out_dir = os.path.join(current_dir, out_dir_rel)
    os.makedirs(out_dir, exist_ok=True)
    output_extended_folder_dat = os.path.join(out_dir, 'extended')
    os.makedirs(output_extended_folder_dat, exist_ok=True)

    # Derived diagnostics used in figures
    resid = conservation_residual(res["N"], res["rho_ent"], res["w"])
    z2 = stability_z2(res["N"], res["a"], res["eps"], P.c_s_scalar)
    target_consistency = P.c_s_scalar / (1.0 + 2.0 * res["nbar"])

    _write_csv(os.path.join(out_dir, "N_conservation_residual.csv"),
               {"N": res["N"], "residual": resid})

    _write_csv(os.path.join(out_dir, "k_vs_nbar.csv"),
               {"k_code": res["ks"], "nbar": res["nbar"]})

    _write_csv(os.path.join(out_dir, "k_vs_Nstar.csv"),
               {"k_code": res["ks"], "N_star": res["N_star"]})

    _write_csv(os.path.join(out_dir, "k_vs_consistency_ratio.csv"),
               {"k_code": res["ks"], "r_over_minus8nt": res["ratio"]})

    _write_csv(os.path.join(out_dir, "N_z2_MukhanovSasaki.csv"),
               {"N": res["N"], "z2": z2})

    _write_csv(os.path.join(out_dir, "k_vs_ring_down_damping.csv"),
               {"k_code": res["ks"], "ring_damp": res["ring_damp"]})

    _write_csv(os.path.join(out_dir, "k_vs_Pzeta_ring.csv"),
               {"k_code": res["ks"], "P_zeta_ring": res["Pz_ring"]})

    _write_csv(os.path.join(out_dir, "k_vs_consistency_target.csv"),
               {"k_code": res["ks"], "cs_over_1plus2nbar": target_consistency})

    _write_csv(os.path.join(out_dir, "k_vs_Pt.csv"),
               {"k_code": res["ks"], "P_t": res["Pt"]})

    _write_csv(os.path.join(out_dir, "k_vs_n_t.csv"),
               {"k_code": res["ks"], "n_t": res["n_t"]})





    # CSV files – one per figure
    _write_csv(os.path.join(output_extended_folder_dat, "N_Hubble_rate.csv"),
               {"N": res["N"], "H_over_H0": res["H"]})  # H already includes H0

    _write_csv(os.path.join(output_extended_folder_dat, "N_w_ent.csv"),
               {"N": res["N"], "w_ent": res["w"]})

    _write_csv(os.path.join(output_extended_folder_dat, "N_rho_ent.csv"),
               {"N": res["N"], "rho_ent_over_rhoc0": res["rho_ent"]})

    _write_csv(os.path.join(output_extended_folder_dat, "N_epsH.csv"),
               {"N": res["N"], "eps_H": res["eps"]})

    _write_csv(os.path.join(output_extended_folder_dat, "N_conformal_time_eta.csv"),
               {"N": res["N"], "eta": res["eta"]})

    _write_csv(os.path.join(output_extended_folder_dat, "k_vs_Pzeta_no_ring.csv"),
               {"k_code": res["ks"], "P_zeta": res["Pz"]})

    _write_csv(os.path.join(output_extended_folder_dat, "k_vs_r.csv"),
               {"k_code": res["ks"], "r": res["r"]})

    _write_csv(os.path.join(output_extended_folder_dat, "k_vs_scalar_tilt.csv"),
               {"k_code": res["ks"], "n_s_minus_1": res["n_s_minus_1"]})

    # Diagnostics and metadata
    diagnostics = {
        "z2_min": float(np.min(z2)),
        "cs2": float(P.c_s_scalar ** 2),
        "pivot_k_phys_Mpc^-1": 0.05,
        "pivot_k_code": float(k_phys_to_code(0.05)),
        "units": {
            "k_code_to_k_phys_Mpc^-1": "k_phys = (H0/c) * k_code",
            "H_plotted": "H_over_H0 (code units)",
            "rho_ent": "dimensionless fraction of critical density (rho_ent/rho_c0)"
        }
    }
    with open(os.path.join(out_dir, "diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    params_dict = {k: getattr(P, k) for k in P.__dict__.keys() if not k.startswith("_")}
    with open(os.path.join(out_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params_dict, f, indent=2)

    # NPZ bundle
    np.savez(os.path.join(out_dir, "experimental_data_v411.npz"),
             # background
             N=res["N"], a=res["a"], H_over_H0=res["H"], eps_H=res["eps"],
             w_ent=res["w"], rho_ent_over_rhoc0=res["rho_ent"], eta=res["eta"],
             # k-dependent
             k_code=res["ks"], N_star=res["N_star"], nbar=res["nbar"],
             P_zeta=res["Pz"], P_zeta_ring=res["Pz_ring"], P_t=res["Pt"], r=res["r"],
             n_t=res["n_t"], consistency_ratio=res["ratio"],
             n_s_minus_1=res["n_s_minus_1"], ring_damp=res["ring_damp"],
             consistency_target=target_consistency,
             # derived diagnostics
             conservation_residual=resid, z2=z2,
             # parameters (as JSON string for portability)
             params_json=json.dumps(params_dict),
             diagnostics_json=json.dumps(diagnostics)
             )

    # Optional: MANIFEST to review CSV feeding figure in src/experimental_data/plots/*
    manifest_lines = [
        "# Manifest: CSV -> figure mapping",
        "N_Hubble_rate.csv -> Hubble_rate.png",
        "N_w_ent.csv -> Entanglement_Equation_of_State.png",
        "N_rho_ent.csv -> Entanglement_Energy_Density.png",
        "N_conservation_residual.csv -> Conservation_Residual.png",
        "N_epsH.csv -> Hubble_Slow_Roll_Parameter.png",
        "N_z2_MukhanovSasaki.csv -> Mukhanov_Sasaki_Variable.png",
        "N_conformal_time_eta.csv -> Conformal_Time.png",
        "k_vs_Nstar.csv -> Freeze_out_e_fold_N.png",
        "k_vs_nbar.csv -> Decoherence_occupancy_profile.png",
        "k_vs_Pzeta_ring.csv -> Scalar_Power_Spectrum.png",
        "k_vs_Pzeta_no_ring.csv -> Post_Act_Occupation_Number_Entanglement-Enhanced Occupation.png",
        "k_vs_Pt.csv -> Tensor_Power_Spectrum.png",
        "k_vs_r.csv -> Tensor_to_Scalar_Ratio.png",
        "k_vs_n_t.csv -> Tensor_tilt_n_t.png",
        "k_vs_consistency_ratio.csv -> Generalized_consistency_ratio.png",
        "k_vs_scalar_tilt.csv -> Scalar_tilt_finite_difference.png",
        "k_vs_ring_down_damping.csv -> Ring_down_damping_factor.png",
        "k_vs_consistency_target.csv -> Target_curve_for_generalized_consistency.png",
        "all_arrays_v410.npz -> NPZ bundle with all arrays + params + diagnostics"
    ]

    with open(os.path.join(out_dir, "MANIFEST.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines))

    return manifest_lines
