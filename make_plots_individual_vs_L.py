import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from hamiltonians import spacing_and_r

from utils import eth_window_variance_deltaE

# arguments: --input_folder
args = argparse.ArgumentParser(description="Make plots from batch run results.")
args.add_argument('--input_folders', "-in", type=str, required=True,
                  help='Comma-separated folders with pickle files (outputs of batch_run).')
parsed_args = args.parse_args()

param_choices = [(0, 0), (2, 2)]

# Make plots for these param choices
def params_match(params_list, params):
    if not params_list:
        return True
    for p in params_list:
        if p == params:
            return True
    return False

length_to_color = {
    10: "#2C7BB6",  # deep blue
    12: "#5AA9D6",  # medium blue
    13: "#F08A4B",  # orange
    14: "#D73027"   # strong red
}

def main():
    # ----- Load data -----
    fig, ax = plt.subplots(4, 1, figsize=(6, 18)) # One for each operator: T, Z, Sz_local, SzSz_local
    fig_var, ax_var = plt.subplots(2, 2, figsize=(7, 7)) # For variance vs energy
    for input_path in parsed_args.input_folders.split(","):
        # Go through all .pkl or .pb files in the input folder
        all_results = []
        for fname in os.listdir(input_path):
            if fname.endswith('.pkl') or fname.endswith('.pb'):
                full_path = os.path.join(input_path, fname)
                with open(full_path, 'rb') as f:
                    data = pickle.load(f)
                    all_results.extend(data)
        if "J2" in all_results[0]:
            print("Hamiltonian type: xxz")
            param1 = "J2"
            param2 = "h_rand"
        else:
            print("Hamiltonian type: xxz_v1")
            param1 = "delta"
            param2 = "theta"
        J2_values = sorted(set(entry[param1] for entry in all_results))
        h_values = sorted(set(entry[param2] for entry in all_results))
        J2_to_idx = {J2: i for i, J2 in enumerate(J2_values)}
        h_to_idx = {h: j for j, h in enumerate(h_values)}
        nJ = len(J2_values)
        nH = len(h_values)
        # Grid of <r>; initialize with NaN in case some combos are missing
        r_grid = np.full((nH, nJ), np.nan, dtype=float)
        # Make an output directory next to the input file
        # ----- Loop over all results, compute spacing stats and histograms -----
        for entry in all_results:
            J2 = entry[param1]
            h_rand = entry[param2]
            if not params_match(param_choices, (J2, h_rand)):
                continue
            # plot each operator's expectation value vs. energy on the subplots
            operators = ["T", "Z", "Sz_local", "SzSz_local"]
            for i, op in enumerate(operators):
                evals = entry["result"]["E"]
                exp_vals = entry["result"][op]
                ax[i].scatter(evals, exp_vals, s=1, alpha=0.5, label=f"{param1}={J2}, {param2}={h_rand} L={entry['L']}")
                ax[i].set_xlabel("Energy of state i")
                ax[i].set_ylabel(f"<i|{op}|i>")
                #energy_window_mean, energy_window_var = eth_window_variance_deltaE(evals, exp_vals, deltaE=1.0)
            for i, op in enumerate(["T", "Z"]):
                evals = entry["result"]["E"]
                exp_vals = entry["result"][op]
                energy_window_mean, energy_window_var = eth_window_variance_deltaE(evals, exp_vals, deltaE=1.0)
                #print("param1 param2", param1, param2)
                if h_rand == 0.0 and J2 == 0.0:
                    ax_var[i, 0].plot(energy_window_mean, energy_window_var, marker='o', linestyle='-', markersize=3,
                                      label=f"L={entry['L']}", color=length_to_color.get(entry['L'], None))
                    ax_var[i, 0].set_title(f"Variance of {op} (Integrable)")
                else:
                    ax_var[i, 1].plot(energy_window_mean, energy_window_var, marker='o', linestyle='-', markersize=3,
                                      label=f"L={entry['L']}", color=length_to_color.get(entry['L'], None))
                    ax_var[i, 1].set_title(f"Variance of {op} (Non-Integrable)")
        for i in range(len(ax)):
            ax[i].legend()
        for i in range(2):
            for j in range(2):
                ax_var[i, j].set_xlabel("E")
                ax_var[i, j].set_ylabel(r"$\sigma^2_E$")
                ax_var[i, j].legend()
                ax_var[i, j].grid(True)
    fig.tight_layout()
    fig.savefig("operator_expectations_vs_energy_varying_L.pdf")
    fig_var.tight_layout()
    fig_var.savefig("operator_variance_vs_energy_varying_L.pdf")
if __name__ == "__main__":
    main()

# python3 make_plots_individual_vs_L.py -in from_remote/L10_SB_small,from_remote/L12_SB_small,from_remote/L14_SB_small,from_remote/L15_SB_small
