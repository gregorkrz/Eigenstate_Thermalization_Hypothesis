
from operators import local_szsz_operator, local_sz_operator, T_operator, Z_operator
from hamiltonians import xxz_hamiltonian, EDParams, diagonalize_full, diag_expectations,xxz_inhomogeneous_hamiltonian_pauli
import numpy as np
from time import time

def run_xxz_hamiltonian_case(L, delta=2.0, J2=0.0, h=0.0, target_magnetization=0, seed=42, break_symmetry_site=-1):
    # Returns: list of eigenvalues; dict of operators to expectation values for each eigenstate
    params = EDParams(
        L=L,
        Delta=delta,
        J2=J2,
        h_rand=h,
        target_magnetization=target_magnetization,
        seed=seed, # seed for the magnetic field
        make_plots=False,
        break_sym_site=break_symmetry_site
    )
    H, basis = xxz_hamiltonian(params)
    # non-diagonal operators
    T = T_operator(params.L, basis)
    # diagonal operators
    Z = Z_operator(params.L, basis)
    SzSz_local = local_szsz_operator(params.L, basis, site=(params.L // 2 - 1))
    Sz_local = local_sz_operator(params.L, basis, site=(params.L // 2))

    print(f"Running ED for H ({H.shape}) with parameters:", params)

    t_start = time()
    evals, evecs = diagonalize_full(H)
    t_diag = time()
    result = {}
    result["E"] = evals
    diag_T = np.real(np.diag(evecs.conj().T @ T @ evecs))
    diag_Z = diag_expectations(evecs, Z)
    diag_Sz = diag_expectations(evecs, Sz_local)
    diag_SzSz = diag_expectations(evecs, SzSz_local)
    result["T"] = diag_T
    result["Z"] = diag_Z
    result["Sz_local"] = diag_Sz
    result["SzSz_local"] = diag_SzSz
    t_end = time()
    print(f"Total time: {t_end - t_start:.2f}. Diagonalization took {t_diag - t_start:.2f} s; computing expectation values took {t_end - t_diag:.2f} s.")
    result["params"] = params
    return result


def run_xxz_hamiltonian_V1_case(L, delta=1.0, theta=0.0, target_magnetization=0, seed=42):
    # Returns: list of eigenvalues; dict of operators to expectation values for each eigenstate
    params = EDParams(
        L=L,
        Delta=delta,
        Theta=theta,
        target_magnetization=target_magnetization,
        seed=seed, # seed for the magnetic field
        make_plots=False
    )
    H, basis = xxz_hamiltonian(params)
    # non-diagonal operators
    T = T_operator(params.L, basis)
    # diagonal operators
    Z = Z_operator(params.L, basis)
    SzSz_local = local_szsz_operator(params.L, basis, site=(params.L // 2 - 1))
    Sz_local = local_sz_operator(params.L, basis, site=(params.L // 2))

    print(f"Running ED for H ({H.shape}) with parameters:", params)

    t_start = time()
    evals, evecs = diagonalize_full(H)
    t_diag = time()
    result = {}
    result["E"] = evals
    diag_T = np.real(np.diag(evecs.conj().T @ T @ evecs))
    diag_Z = np.real(np.diag(evecs.conj().T @ Z @ evecs))
    diag_Sz = np.real(np.diag(evecs.conj().T @ Sz_local @ evecs))
    diag_SzSz = np.real(np.diag(evecs.conj().T @ SzSz_local @ evecs))
    #diag_Z = diag_expectations(evecs, Z)
    #diag_Sz = diag_expectations(evecs, Sz_local)
    #diag_SzSz = diag_expectations(evecs, SzSz_local)
    result["T"] = diag_T
    result["Z"] = diag_Z
    result["Sz_local"] = diag_Sz
    result["SzSz_local"] = diag_SzSz
    t_end = time()
    print(f"Total time: {t_end - t_start:.2f}. Diagonalization took {t_diag - t_start:.2f} s; computing expectation values took {t_end - t_diag:.2f} s.")
    result["params"] = params
    return result

