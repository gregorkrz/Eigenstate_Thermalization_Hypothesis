import numpy as np
import itertools
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from numpy.linalg import eigh
from utils import bitstring_to_int, state_vector_from_bitstring
import random

# ---------------- Configuration ----------------
@dataclass
class EDParams:
    L: int = 12              # Chain length
    J: float = 1.0           # NN exchange (sets energy unit)
    Delta: float = 1.0       # XXZ anisotropy
    J2: float = 0.0          # NNN Ising coupling (breaks integrability when != 0)
    h_rand: float = 0.0      # Random longitudinal field strength in [-h, h]
    target_magnetization: int = 0  # total Sz sector: M = L//2 + target_magnetization
    seed: int = 7            # RNG seed for random fields
    observable_site: int = None    # Site for local Sz; default = L//2
    n_windows: int = 20      # Energy windows for ETH variance
    make_plots: bool = False
    Theta: float = 0.0 # Only used for the XXZ hamiltonian V2
    break_sym_site: float = -1 # Site idx at which we will introduce a tiny field of 0.001 to break reflection symmetry; -1 means no symmetry breaking

# --------------- Basis: fixed Sz sector ---------------
def build_basis_fixed_magnetization(L, M):
    """Return list of bitstrings with exactly M up-spins and a dict to indices."""
    basis = []
    for bits in itertools.combinations(range(L), M):
        s = 0
        for b in bits:
            s |= (1 << b)
        basis.append(s)
    index = {state: i for i, state in enumerate(basis)}
    return basis, index


def xxz_hamiltonian(params: EDParams): # Open BC
    L, J, Delta, J2, h = params.L, params.J, params.Delta, params.J2, params.h_rand
    rng = np.random.default_rng(params.seed)
    if params.observable_site is None:
        params.observable_site = L // 2
    M = L // 2 + params.target_magnetization
    if not (0 <= M <= L):
        raise ValueError("Invalid target magnetization for given L.")
    basis, index = build_basis_fixed_magnetization(L, M)
    dim = len(basis)
    rows, cols, data = [], [], []
    diag = np.zeros(dim, dtype=np.float64)
    hz = rng.uniform(-h, h, size=L) if h > 0 else np.zeros(L)
    if params.break_sym_site >= 0 and params.break_sym_site < L:
        hz[params.break_sym_site] += 0.1  # Tiny field to break reflection symmetry
        print("Applied tiny symmetry-breaking field at site", params.break_sym_site)
    def sz_at(state, i):  # returns ±1 (σ^z eigenvalue)
        return 1.0 if (state >> i) & 1 else -1.0
    # ----- Diagonal terms: NN Ising (OBC) + random hz + NNN Ising (OBC) -----

    for k, state in enumerate(basis):
        d = 0.0
        # NN σ^z σ^z (OBC)
        for i in range(L - 1):
            j = i + 1
            d += J * Delta * 0.25 * sz_at(state, i) * sz_at(state, j)
        # random longitudinal field (doesn't break Sz conservation)
        if h > 0 or params.break_sym_site >= 0:
            for i in range(L):
                d += 0.5 * hz[i] * sz_at(state, i)
        # NNN σ^z σ^z (OBC)
        if J2 != 0.0:
            for i in range(L - 2):
                j = i + 2
                si = 0.5 if (state >> i) & 1 else -0.5
                sj = 0.5 if (state >> j) & 1 else -0.5
                d += J2 * (4.0 * si * sj) * 0.25  # = J2 * S^z_i S^z_{i+2}
        diag[k] += d

    # ----- Off-diagonal NN flip-flop (OBC) -----
    for k, state in enumerate(basis):
        for i in range(L - 1): # NO wrap bond
            j = i + 1
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi != bj:
                new_state = state ^ ((1 << i) | (1 << j))
                new_idx = index.get(new_state, None)
                if new_idx is not None:
                    rows.append(k); cols.append(new_idx); data.append(0.5 * J)

    # ----- Next-nearest-neighbor XY flip-flop (OBC) -----
    J2_perp = params.J2  # Or make a separate param if you want J2z different
    for k, state in enumerate(basis):
        for i in range(L - 2): # No wrap
            j = i + 2
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi != bj:
                # flip i and j, leave the middle spin untouched
                new_state = state ^ ((1 << i) | (1 << j))
                new_idx = index.get(new_state, None)
                if new_idx is not None:
                    rows.append(k); cols.append(new_idx); data.append(0.5 * J2_perp)
    from scipy.sparse import coo_matrix
    H = coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64)
    H = H + H.T
    H = H + coo_matrix((diag, (np.arange(dim), np.arange(dim))), shape=(dim, dim))
    return H.toarray(), basis

# ----------------- Observables & diagnostics -----------------
def local_sz_operator(L, basis, site):
    diag = np.array([0.5 if (state >> site) & 1 else -0.5 for state in basis], dtype=np.float64)
    return np.diag(diag)

def diagonalize_full(H_dense):
    evals, evecs = eigh(H_dense) # Full spectrum & eigenvectors (dim ~ binomial(L, L/2))
    return evals, evecs

def spacing_and_r(evals, tol=1e-5, frac_center=0.7):
    E = np.sort(evals)
    m = int((1 - frac_center) * len(E) / 2)
    E = E[m:len(E)- m]
    spacings = np.diff(E)
    spacings = spacings[spacings > tol]
    s1, s2 = spacings[:-1], spacings[1:]
    r = np.minimum(s1, s2) / np.maximum(s1, s2)
    return spacings, r, float(np.mean(r))

def diag_expectations(evecs, Op):
    # Operator is diagonal in our working basis → <psi|O|psi> = sum_i |psi_i|^2 O_ii
    diagO = np.diag(Op)
    weights = (np.abs(evecs)**2).T  # Shape: (n_eigs, dim)
    return weights @ diagO



def xxz_inhomogeneous_hamiltonian_pauli(params: EDParams):
    """
    Spin-1/2 XXZ chain with *inhomogeneous* zz couplings (Pauli normalization) and OBC:

        H = sum_{i=1}^{L-1} [ J (σ_i^x σ_{i+1}^x + σ_i^y σ_{i+1}^y) + Δ_i σ_i^z σ_{i+1}^z ],

    where
        Δ_i = Δ + θ (2i - L)/(L - 2),   i = 1,...,L-1,

    implemented in the fixed-Sz (σ^z) computational basis.
    """

    L, J, Delta0, theta = params.L, params.J, params.Delta, params.theta
    if params.observable_site is None:
        params.observable_site = L // 2

    # fixed total S^z sector
    M = L // 2 + params.target_magnetization
    if not (0 <= M <= L):
        raise ValueError("Invalid target_magnetization for given L.")
    basis, index = build_basis_fixed_magnetization(L, M)
    dim = len(basis)

    rows, cols, data = [], [], []
    diag = np.zeros(dim, dtype=np.float64)

    # Δ_i on bonds i=0..L-2 (code index), corresponding to paper's i=1..L-1
    bond_Delta = np.zeros(L - 1, dtype=np.float64)
    for i in range(L - 1):
        ip = i + 1  # 1-based index like in the paper
        bond_Delta[i] = Delta0 + theta * (2 * ip - L) / (L - 2)

    def sigma_z(state, site):   # eigenvalue of σ^z at 'site' (±1)
        return 1.0 if ((state >> site) & 1) else -1.0

    # ----- Diagonal term: Δ_i σ_i^z σ_{i+1}^z -----
    for k, state in enumerate(basis):
        d = 0.0
        for i in range(L - 1):   # OBC: bonds (i, i+1)
            j = i + 1
            d += bond_Delta[i] * sigma_z(state, i) * sigma_z(state, j)
        diag[k] += d

    # ----- Off-diagonal term: J(σ^x σ^x + σ^y σ^y) -----
    # In the {|↑↓>,|↓↑>} subspace, this operator is:
    #   ( 0  2 )
    #   ( 2  0 )
    # so the off-diagonal matrix element between |↑↓> and |↓↑> is 2J.
    for k, state in enumerate(basis):
        for i in range(L - 1):   # OBC
            j = i + 1
            si = (state >> i) & 1
            sj = (state >> j) & 1
            if si != sj:
                new_state = state ^ ((1 << i) | (1 << j))  # flip i and j
                new_idx = index.get(new_state, None)
                if new_idx is not None:
                    rows.append(k); cols.append(new_idx); data.append(2.0 * J)

    H = coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.float64)
    H = H + H.T
    H = H + coo_matrix((diag, (np.arange(dim), np.arange(dim))), shape=(dim, dim))
    return H.toarray(), basis


def make_product_state(L, basis, kind="neel", site_offset=0):
    """
    kind ∈ {"neel","domain","custom", "random"}.
    - "neel": ...0101 pattern (offset toggles the pattern)
    - "domain": left half up, right half down
    - "custom": pass integer bitstring via site_offset
    - "random": randomly space out L/2 up-spins
    Up = 1, Down = 0; site 0 is least-significant bit.
    """
    if kind == "neel":
        bits = [((i + site_offset) % 2) for i in range(L)]
        return state_vector_from_bitstring(basis, bitstring_to_int(bits))
    elif kind == "domain":
        bits = [1 if i < L//2 else 0 for i in range(L)]
        print("bits:", bits)
        return state_vector_from_bitstring(basis, bitstring_to_int(bits))
    elif kind == "custom":
        return state_vector_from_bitstring(basis, site_offset)
    elif kind == "random":
        positions = random.sample(range(L), L//2)
        bits = [1 if i in positions else 0 for i in range(L)]
        return state_vector_from_bitstring(basis, bitstring_to_int(bits))
    else:
        raise ValueError("kind must be 'neel', 'domain', or 'custom'.")

