import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from hamiltonians import spacing_and_r

# arguments: --input_folder
args = argparse.ArgumentParser(description="Make plots from batch run results.")
args.add_argument('--input_folder', "-in", type=str, required=True,
                  help='Folder with pickle files (outputs of batch_run).')
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

def main():
    # ----- Load data -----
    fig, ax = plt.subplots(4, 1, figsize=(4, 10)) # One for each operator: T, Z, Sz_local, SzSz_local
    input_path = parsed_args.input_folder
    # go through all .pkl or .pb files in the input folder
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
    out_dir = os.path.splitext(os.path.basename(input_path))[0] + "_plots"
    os.makedirs(out_dir, exist_ok=True)

    # ----- Loop over all results, compute spacing stats and histograms -----
    for entry in all_results:
        J2 = entry[param1]
        h_rand = entry[param2]
        if not params_match(param_choices, (J2, h_rand)):
            continue
        # plot each operator's expectation value vs. energy on the subplots
        operators = ["T", "Z", "Sz_local", "SzSz_local"]
        operators_tex = ["T", "Z", "S^z_{L/2}", "S^z_{L/2} S^z{L/2+1}"]
        for i, op in enumerate(operators):
            evals = entry["result"]["E"]
            exp_vals = entry["result"][op]
            ax[i].scatter(evals, exp_vals, s=1, alpha=0.5, label=f"$J_2$={J2}, $h_{{rand}}$={h_rand}")
            ax[i].set_xlabel("$E_i$")
            #ax[i].set_ylabel(f"<i|{op}|i>")
            ax[i].set_ylabel(f"$\\langle i|{operators_tex[i]}|i \\rangle$")
    for i in range(len(ax)):
        ax[i].legend()
        ax[i].grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "operator_expectations_vs_energy.png"), dpi=500)


if __name__ == "__main__":
    main()

