import argparse

job_template = """#!/bin/bash
#SBATCH --partition=PARTITION_HERE               # Specify the partition
#SBATCH --account=ACCOUNT_HERE                  # Specify the account
#SBATCH --mem={mem}                      # Request 60GB of memory and 1 CPU
#SBATCH --cpus-per-task=2                # Request 4 CPU cores
#SBATCH --nodes=1                        # Request 1 node
#SBATCH --time=02:00:00                  # Set the time limit to 12 hrs. - this times out for 10k events!!!!
#SBATCH --job-name={job_name}            # Name the job
#SBATCH --output=OUTPUT_DIR_HERE/{out_filename}       # Redirect stdout to a log file
#SBATCH --error=OUTPUT_DIR_HERE/{err_filename}        # Redirect stderr to a log file

export APPTAINER_CACHEDIR=APPTAINER_CACHEDIR_HERE
export APPTAINER_TMPDIR=APPTAINER_TMPDIR_HERE

singularity exec -B / --nv docker://gkrz/alma:v0 /bin/bash -lc 'source /cvmfs/fcc.cern.ch/sw/latest/setup.sh && {cmd}'
1
"""

# Args: output_dir, N_jobs_split, L, J2_min, J2_max, h_min, h_max, N_J2, N_h
parser = argparse.ArgumentParser(description="Generate SLURM job scripts for batch ED runs.")
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save generated job scripts.')
parser.add_argument('--N_jobs_split', type=int, default=1, help='Number of jobs to split the parameter grid into.')
parser.add_argument('--L', type=int, required=True, help='System size L.')
parser.add_argument('--J2_min', type=float, default=0.0, help='Minimum J2 value.')
parser.add_argument('--J2_max', type=float, default=1.0, help='Maximum J2 value.')
parser.add_argument('--h_min', type=float, default=0.0, help='Minimum h_rand value.')
parser.add_argument('--h_max', type=float, default=1.0, help='Maximum h_rand value.')
parser.add_argument('--N_J2', type=int, default=5, help='Number of J2 values.')
parser.add_argument('--N_h', type=int, default=5, help='Number of h_rand values.')
parser.add_argument("--SB", action='store_true', help="Whether to include symmetry breaking (small field 0.1 at site 0).")
parser.add_argument("--mem", type=str, default="10000", help="Memory per job (e.g., 60000 for 60GB).")
parser.add_argument("--run", action='store_true', help="Whether to submit the jobs after generating the scripts.")

parsed_args = parser.parse_args()

# Split the J2 scan into N_jobs_split parts. Each job needs to be passed a list of J2 values and h values and will do a grid scan using the provided values.

import numpy as np
J2_values = np.linspace(parsed_args.J2_min, parsed_args.J2_max, parsed_args.N_J2)
h_values = np.linspace(parsed_args.h_min, parsed_args.h_max, parsed_args.N_h)
J2_splits = np.array_split(J2_values, parsed_args.N_jobs_split)
# the command is python- -output_file L13_small_scan/out_<JOB_ID>.pb --L 13 --J2 0 1 2 3 4 5 --h_rand 0 0.1 0.2 ...
import os
os.makedirs(parsed_args.output_dir, exist_ok=True)
for job_id, J2_subset in enumerate(J2_splits):
    J2_list_str = ' '.join(f"{J2:.6f}" for J2 in J2_subset)
    h_list_str = ' '.join(f"{h:.6f}" for h in h_values)
    output_file = os.path.join(parsed_args.output_dir, f"L{parsed_args.L}_scan_job{job_id}.pb")
    sb_suffix = ""
    if parsed_args.SB:
        sb_suffix = "--break-symmetry-site 0"
    cmd = f"python3 batch_run.py --output_file {output_file} --L {parsed_args.L} --J2 {J2_list_str} --h_rand {h_list_str} {sb_suffix}"
    job_name = f"ED_L{parsed_args.L}_job{job_id}"
    out_filename = f"ED_L{parsed_args.L}_job{job_id}.out"
    err_filename = f"ED_L{parsed_args.L}_job{job_id}.err"
    job_script_content = job_template.format(
        job_name=job_name,
        out_filename=out_filename,
        err_filename=err_filename,
        cmd=cmd,
        mem=parsed_args.mem
    )
    job_script_path = os.path.join(parsed_args.output_dir, f"job_script_{job_id}.sh")
    with open(job_script_path, 'w') as f:
        f.write(job_script_content)
    print(f"Generated job script: {job_script_path}")
    if parsed_args.run:
        os.system(f"sbatch {job_script_path}")
        print(f"Submitted job script: {job_script_path}")


# Parameter scans:
# python generate_slurm_job.py --output_dir L15_SB --N_jobs_split 10 --L 15 --J2_min 0.0 --J2_max 10.0 --h_min 0.0 --h_max 10.0 --N_J2 25 --N_h 25 --SB --mem 35000
# python generate_slurm_job.py --output_dir L15_SB_HigherRes --N_jobs_split 10 --L 15 --J2_min 0.0 --J2_max 1.0 --h_min 0.0 --h_max 1.0 --N_J2 25 --N_h 25 --SB --mem 35000

