import numpy as np

def eth_window_variance(evals, diag_O, n_windows=20):
    order = np.argsort(evals)
    E, O = evals[order], diag_O[order]
    N = len(E)
    w = max(1, N // n_windows)
    centers, vars_ = [], []
    for k in range(0, N - w + 1, w):
        Ei = E[k:k+w]
        Oi = O[k:k+w]
        centers.append(np.mean(Ei)); vars_.append(np.var(Oi))
    return np.array(centers), np.array(vars_)


def eth_window_variance_deltaE(evals, diag_O, deltaE=1.0):
    """
    Compute variance of diag_O in energy windows of width `deltaE`.
    Returns window centers and variances.
    """
    order = np.argsort(evals)
    E, O = evals[order], diag_O[order]
    if deltaE <= 0:
        raise ValueError("deltaE must be positive")
    #minE, maxE = E.min(), E.max()
    minE = -10.0
    maxE = 10.0
    bins = np.arange(minE, maxE + deltaE, deltaE)
    centers, vars_ = [], []
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            mask = (E >= bins[i]) & (E <= bins[i + 1])
        else:
            mask = (E >= bins[i]) & (E < bins[i + 1])
        if not mask.any():
            continue
        centers.append(np.mean(E[mask]))
        vars_.append(np.var(O[mask]))
    return np.array(centers), np.array(vars_)


def bin_eigenval_diff(eigenvals):
    # Compute histogram of level spacings
    # E_i - E_{i-1}
    spacings = np.abs(np.diff(eigenvals))
    hist, bin_edges = np.histogram(spacings, bins=np.linspace(0, 0.05, 40), density=True)
    return hist, bin_edges

def state_vector_from_bitstring(basis, bitstring_int: int):
    dim = len(basis)
    vec = np.zeros(dim, dtype=np.complex128)
    try:
        idx = basis.index(bitstring_int)
    except ValueError:
        raise ValueError("Chosen bitstring is not in the current fixed-Sz sector.")
    vec[idx] = 1.0
    return vec

def bitstring_to_int(bits):
    s = 0
    for i, b in enumerate(bits):
        if b not in (0, 1):
            raise ValueError("bitstring must be only 0/1")
        if b: s |= (1 << i)
    return s

