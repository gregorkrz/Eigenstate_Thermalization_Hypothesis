import numpy as np

def local_szsz_operator(L, basis, site):
    j = site + 1
    if j >= L:
        raise ValueError("site+1 exceeds chain length for SzSz with OBC.")
    diag = np.zeros(len(basis), dtype=np.float64)
    for k, state in enumerate(basis):
        si = 0.5 if ((state >> site) & 1) else -0.5
        sj = 0.5 if ((state >> j) & 1) else -0.5
        diag[k] = si * sj
    return np.diag(diag)

def local_sz_operator(L, basis, site):
    diag = np.array([0.5 if ((state >> site) & 1) else -0.5 for state in basis], dtype=np.float64)
    return np.diag(diag)


def T_operator(L, basis):
    dim = len(basis)
    rows, cols, data = [], [], []

    for k, state in enumerate(basis):
        for i in range(L - 2):  # no wrap
            j = i + 2
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            # Flip i and j if they are opposite
            if bi != bj:
                new_state = state ^ ((1 << i) | (1 << j))
                # new_state might not lie in the same fixed-Sz sector — check:
                try:
                    new_idx = basis.index(new_state)
                except ValueError:
                    continue
                # Matrix element = 1 (since σ^xσ^x + σ^yσ^y = 2(S^+S^- + S^-S^+))
                rows.append(k)
                cols.append(new_idx)
                data.append(1.0)  # We'll divide by L later
    # Build matrix and normalize
    from scipy.sparse import coo_matrix
    T = coo_matrix((data, (rows, cols)), shape=(dim, dim)).toarray()
    return T / L


def Z_operator(L, basis, Delta_tilde=None):
    dim = len(basis)
    # Set up Δ̃_i for i = 0..L-2
    if Delta_tilde is None:
        D = np.ones(L - 1, dtype=np.float64)
    elif np.isscalar(Delta_tilde):
        D = float(Delta_tilde) * np.ones(L - 1, dtype=np.float64)
    else:
        D = np.asarray(Delta_tilde, dtype=np.float64)
        if D.shape[0] != L - 1:
            raise ValueError(f"Delta_tilde must have length L-1 = {L-1}, got {D.shape[0]}")
    diag = np.zeros(dim, dtype=np.float64)
    for k, state in enumerate(basis):
        val = 0.0
        for i in range(L - 1):
            # σ^z eigenvalues ±1 in your convention
            szi  =  1.0 if ((state >> i)   & 1) else -1.0
            szip =  1.0 if ((state >> (i+1)) & 1) else -1.0
            val += D[i] * szi * szip
        diag[k] = val / L   # Overall: 1/N normalization (here N=L)
    return np.diag(diag)

