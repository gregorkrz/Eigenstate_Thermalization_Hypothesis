
## Detailed plots with a larger system size
To reproduce detailed plots with L=16:

First run the following commands to run exact diagonalization on a system of size L=16:
```
python batch_run.py --output_file L16/out.pb --L 16 --J2 0.0 2.0 --h_rand 0.0 2.0 
python batch_run.py --output_file L16_SB/out.pb --L 16 --break-symmetry-site 0 --J2 0.0 2.0 --h_rand 0.0 2.0 
```

Then, make the plots:
```
python make_plots.py -in L16
python make_plots.py -in L16_SB
python make_plots_individual.py -in L16
python make_plots_individual.py -in L16_SB
```
## Detailed parameter scan plots

To also produce the plots with a smaller lattice size but a larger parameter scan, see the `generate_slurm_job.py` script.

## Plots of expectation value variance with varying system size L

To produce the results with varying system size L, run
```
python batch_run.py --output_file L10_SB/out.pb --L 10 --break-symmetry-site 0 --J2 0.0 2.0 --h_rand 0.0 2.0 
python batch_run.py --output_file L11_SB/out.pb --L 11 --break-symmetry-site 0 --J2 0.0 2.0 --h_rand 0.0 2.0 
python batch_run.py --output_file L12_SB/out.pb --L 12 --break-symmetry-site 0 --J2 0.0 2.0 --h_rand 0.0 2.0 
python batch_run.py --output_file L13_SB/out.pb --L 13 --break-symmetry-site 0 --J2 0.0 2.0 --h_rand 0.0 2.0 
python batch_run.py --output_file L14_SB/out.pb --L 14 --break-symmetry-site 0 --J2 0.0 2.0 --h_rand 0.0 2.0 
python batch_run.py --output_file L15_SB/out.pb --L 15 --break-symmetry-site 0 --J2 0.0 2.0 --h_rand 0.0 2.0 
```

as well as the plotting script 
```
python make_plots_individual_vs_L.py -in L10_SB,L11_SB,L12_SB,L13_SB,L14_SB,L15_SB,L16_SB
```

