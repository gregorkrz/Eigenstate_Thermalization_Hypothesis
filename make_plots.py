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


def main():
    # ----- Load data -----
    input_path = parsed_args.input_folder
    # go through all .pkl or .pb files in the input folder
    all_results = []
    for fname in os.listdir(input_path):
        if fname.endswith('.pkl') or fname.endswith('.pb'):
            full_path = os.path.join(input_path, fname)
            with open(full_path, 'rb') as f:
                data = pickle.load(f)
                all_results.extend(data)
    # For the <r> vs J2 at h=0 plots
    data_J2 = []
    data_J2_r_avg = []
    # Expect all_results to be a list of dicts like:
    # {
    #   "J2": J2_value,
    #   "h_rand": h_value,
    #   "L": L,
    #   "result": {
    #       "E": evals,
    #       "T": ...,
    #       "Z": ...,
    #       "Sz_local": ...,
    #       "SzSz_local": ...
    #   },
    #   "Hamiltonian": ...
    # }

    # ----- Collect parameter grids -----
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

    # grid of <r>; initialize with NaN in case some combos are missing
    r_grid = np.full((nH, nJ), np.nan, dtype=float)

    # make an output directory next to the input file
    out_dir = os.path.splitext(os.path.basename(input_path))[0] + "_plots"
    os.makedirs(out_dir, exist_ok=True)

    # ----- Loop over all results, compute spacing stats and histograms -----
    for entry in all_results:
        J2 = entry[param1]
        h_rand = entry[param2]
        evals = np.array(entry["result"]["E"], dtype=float)

        # spacing + r
        spacings, r_vals, r_mean = spacing_and_r(evals)
        print(f"J2={J2:.3g}, h_rand={h_rand:.3g}: <r>={r_mean:.4f}")

        if h_rand == 0.0:
            data_J2.append(J2)
            data_J2_r_avg.append(r_mean)

        # fill the grid
        j_idx = J2_to_idx[J2]
        h_idx = h_to_idx[h_rand]
        r_grid[h_idx, j_idx] = r_mean

        # normalized spacings for histogram
        s_norm = spacings / np.mean(spacings)
        # Use the 60% middle of spectrum
        n_spacings = len(s_norm)
        lower_idx = int(0.2 * n_spacings)
        upper_idx = int(0.8 * n_spacings)
        #s_norm = s_norm[lower_idx:upper_idx]
        #s_norm=spacings
        #s_norm = s_norm[]

        # Histogram of level spacings for this (J2, h_rand)
        if (J2 == 0.0 and h_rand == 0.0) or (J2 == 2.0 and h_rand == 2.0) or (J2==1.0 and h_rand==1.0):
            fig, ax = plt.subplots(figsize=(5, 4))
            bins = np.linspace(0.0, 3.0, 80)
            ax.hist(s_norm, bins=bins, density=True, alpha=0.8, edgecolor='none')
            # plot both the exponential and Wigner-Dyson distributions for comparison
            s_plot = np.linspace(0.0, 3.0, 200)
            L = entry["L"]
            P_exp = np.exp(-s_plot)  # Poisson
            P_WD = (np.pi / 2) * s_plot * np.exp(-(np.pi / 4) * s_plot**2)
            ax.plot(s_plot, P_exp, 'r--', label='Poisson', linewidth=2)
            ax.plot(s_plot, P_WD, 'g--', label='Wigner-Dyson (GOE)', linewidth=2)
            ax.legend()
            #ax.set_yscale("log")
            ax.set_xlabel(r"Normalized spacing $s$")
            ax.set_ylabel(r"$P(s)$")
            ax.set_title(f"Level spacings (L = {L}) \n$J_2$={J2:.3g}, $h_{{rand}}$={h_rand:.3g}, <r>={r_mean:.3f}")
            fname = f"spacing_hist_J2_{J2:.3g}_h_{h_rand:.3g}.pdf"
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, fname), dpi=200)
            plt.close(fig)

    # ----- 2D heatmap of <r> vs (J2, h_rand) using pcolormesh -----

    fig, ax = plt.subplots(figsize=(6, 5))

    # Convert parameter lists to arrays
    J2_arr = np.array(J2_values)
    h_arr = np.array(h_values)

    # Create 2D coordinate grids (centers)
    J2_grid, h_grid = np.meshgrid(J2_arr, h_arr)

    # To use pcolormesh, we need the *edges* of each cell
    # Build edges by extending midpoints
    def _edges(x):
        x = np.array(x)
        mid = (x[:-1] + x[1:]) / 2
        left = x[0] - (mid[0] - x[0])
        right = x[-1] + (x[-1] - mid[-1])
        return np.concatenate([[left], mid, [right]])

    J2_edges = _edges(J2_arr)
    h_edges = _edges(h_arr)

    # pcolormesh expects 2D arrays of shape (nH+1, nJ+1)
    J2_edges_2D, h_edges_2D = np.meshgrid(J2_edges, h_edges)

    # Plot
    cmap = "viridis"
    pm = ax.pcolormesh(
        J2_edges_2D,
        h_edges_2D,
        r_grid,
        shading="auto",
        cmap=cmap,
        vmin=0.3,  # Poisson-ish
        vmax=0.6,  # GOE-ish
    )

    ax.set_xlabel(r"$J_2$")
    ax.set_ylabel(r"$h_{\mathrm{rand}}$")
    ax.set_title(r"Mean $r$ vs $(J_2, h_{\mathrm{rand}})$")

    cbar = fig.colorbar(pm, ax=ax)
    cbar.set_label(r"$\langle r \rangle$")

    fig.tight_layout()
    heatmap_path = os.path.join(out_dir, "r_mean_heatmap_pcolormesh.pdf")
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)

    print(f"Saved <r> heatmap with pcolormesh to: {heatmap_path}")
    print(f"Saved spacing histograms into: {out_dir}")
    fig_J2, ax_J2 = plt.subplots(figsize=(6, 5))  # for <r> vs J2 at fixed h=0
    data_J2 = np.array(data_J2)
    data_J2_r_avg = np.array(data_J2_r_avg)
    # Sort J2
    sort_idx = np.argsort(data_J2)
    data_J2 = data_J2[sort_idx]
    data_J2_r_avg = data_J2_r_avg[sort_idx]
    ax_J2.plot(data_J2, data_J2_r_avg, marker='.', label="L = " + str(all_results[0]["L"]) + " data")
    ax_J2.set_xlabel(r"$J_2$")
    ax_J2.set_ylabel(r"$\langle r \rangle$ at $h_{rand.} = 0$")
    ax_J2.set_title("L = " + str(all_results[0]["L"]))
    # do horizontal dashed black lines at 0.386 and 0.5307
    ax_J2.axhline(0.386, color='black', linestyle='--', label='Poisson')
    ax_J2.axhline(0.53, color='red', linestyle='--', label='Wigner-Dyson (GOE)')
    ax_J2.legend()
    fig_J2.tight_layout()
    fig_J2.savefig(os.path.join(out_dir, "1D_plot_r_mean_vs_J2.pdf"))


if __name__ == "__main__":
    main()
