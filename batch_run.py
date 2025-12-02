import pickle
import argparse
from run_case import run_xxz_hamiltonian_V1_case
from run_case import run_xxz_hamiltonian_case
# Arguments: --output_file; --H_type (xxz or xxz_v1); --L (system size); --J2 (list); --h_rand (list)

args = argparse.ArgumentParser(description="Batch run for Hamiltonian cases.")
args.add_argument('--output_file', type=str, required=True, help='Output file to save results.')
args.add_argument('--hamiltonian', type=str, choices=['xxz', 'xxz_v1'], default="xxz", required=False, help='Type of Hamiltonian to run.')
args.add_argument("--L", type=int, required=True, help='System size L.')
args.add_argument("--J2", type=float, nargs='+', default=[0.0], help='List of J2 values to run.')
args.add_argument("--h_rand", type=float, nargs='+', default=[0.0], help='List of h_rand values to run.')
# now similar to J2 and H_rand, add delta and theta as arguments to be used with xxz_v1
args.add_argument("--delta", type=float, nargs="+", default=[0.0], help='Delta parameter for xxz_v1 Hamiltonian.')
args.add_argument("--theta", type=float, nargs="+", default=[0.0], help='Theta parameter for xxz_v1 Hamiltonian.')

args.add_argument("--break-symmetry-site", type=int, default=-1, help="Site index to apply a tiny field to break reflection symmetry; -1 means no symmetry breaking.")

# If the directory of output_file does not exist, create it
import os
output_dir = os.path.dirname(args.parse_args().output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
# assert that the file does not already exist
if os.path.exists(args.parse_args().output_file):
    raise FileExistsError(f"Output file {args.parse_args().output_file} already exists.")

parsed_args = args.parse_args()

if parsed_args.hamiltonian == "xxz_v1":
    all_results = []
    for d in parsed_args.delta:
        for th in parsed_args.theta:
            print(f"Running case L={parsed_args.L}, delta={d}, theta={th}")
            result = run_xxz_hamiltonian_V1_case(
                L=parsed_args.L,
                delta=d,
                theta=th,
                target_magnetization=0,
                seed=42
            )
            all_results.append({
                "delta": d,
                "theta": th,
                "L": parsed_args.L,
                "result": result,
                "Hamiltonian": parsed_args.hamiltonian
            })

    with open(parsed_args.output_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Saved all results to {parsed_args.output_file}")

elif parsed_args.hamiltonian == "xxz":
    all_results = []
    for J2 in parsed_args.J2:
        for h_rand in parsed_args.h_rand:
            print(f"Running case L={parsed_args.L}, J2={J2}, h_rand={h_rand}")
            result = run_xxz_hamiltonian_case(
                L=parsed_args.L,
                J2=J2,
                h=h_rand,
                delta=1.0,
                target_magnetization=0,
                seed=42,
                break_symmetry_site=parsed_args.break_symmetry_site
            )
            all_results.append({
                "J2": J2,
                "h_rand": h_rand,
                "L": parsed_args.L,
                "result": result,
                "Hamiltonian": parsed_args.hamiltonian
            })

    with open(parsed_args.output_file, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"Saved all results to {parsed_args.output_file}")
