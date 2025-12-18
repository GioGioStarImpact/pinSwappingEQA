from itertools import permutations
from typing import List, Tuple, Dict, Iterable, Union

# -----------------------------
# Helpers: truth table encoding
# -----------------------------

def bits_to_int(bits: Iterable[int]) -> int:
    """bits is LSB-first (bit0 is least significant)."""
    v = 0
    for i, b in enumerate(bits):
        if b not in (0, 1):
            raise ValueError("truth table must be binary (0/1).")
        v |= (b & 1) << i
    return v

def parse_truth_table_rows(N: int, M: int, rows: List[Union[int, str, List[int], Tuple[int, ...]]]) -> List[int]:
    """
    Convert various row formats to int outputs.
    F[u] is an M-bit int in output-pin order (Z1 is bit0, Z2 is bit1, ...).
    rows must have length 2^N, indexed by u (pin input pattern).

    Accepted row formats:
      - int: 0..(2^M-1)
      - str: e.g. "010" meaning Z1=0,Z2=1,Z3=0 (MSB-left). We'll parse left->right as ZM..Z1.
      - list/tuple of ints: [z1,z2,...,zM] (LSB-first in list order)
    """
    if len(rows) != (1 << N):
        raise ValueError(f"Need exactly {1<<N} rows for N={N}, got {len(rows)}")

    F = []
    for r in rows:
        if isinstance(r, int):
            if r < 0 or r >= (1 << M):
                raise ValueError(f"Row int out of range for M={M}: {r}")
            F.append(r)
        elif isinstance(r, str):
            s = r.strip()
            if len(s) != M or any(c not in "01" for c in s):
                raise ValueError(f"Bad row string '{r}', need length M={M} of 0/1")
            # interpret as MSB-left "zM...z1"
            bits = [int(c) for c in reversed(s)]  # now LSB-first [z1..zM]
            F.append(bits_to_int(bits))
        elif isinstance(r, (list, tuple)):
            if len(r) != M:
                raise ValueError(f"Bad row list length, need M={M}, got {len(r)}")
            F.append(bits_to_int(r))  # assume [z1..zM]
        else:
            raise TypeError(f"Unsupported row type: {type(r)}")
    return F

def int_to_bitstr(v: int, width: int) -> str:
    """Return MSB-left bitstring of given width."""
    return "".join("1" if (v >> i) & 1 else "0" for i in reversed(range(width)))


# -----------------------------
# Permutation / mapping logic
# -----------------------------

def inverse_perm(pi: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    pi maps net index i -> pin index pi[i].
    returns inv mapping pin index p -> net index inv[p].
    """
    N = len(pi)
    inv = [None] * N
    for i, p in enumerate(pi):
        inv[p] = i
    if any(x is None for x in inv):
        raise ValueError("pi is not a permutation")
    return tuple(inv)

def net_pattern_to_pin_pattern(t: int, inv_pi: Tuple[int, ...], N: int) -> int:
    """
    t is N-bit pattern for nets (net1..netN) in LSB-first: net1 is bit0, net2 is bit1, ...
    inv_pi[p] tells which net index feeds pin p.
    returns u: N-bit pattern for pins (A1..AN) in LSB-first: A1 is bit0, ...
    """
    u = 0
    for p in range(N):
        net_i = inv_pi[p]
        bit = (t >> net_i) & 1
        u |= bit << p
    return u

def build_output_functions_bitvectors(
    F: List[int],
    N: int,
    M: int,
    pi: Tuple[int, ...],
    verbose: bool = False,
    show_patterns: int = 8
) -> List[int]:
    """
    For a fixed input permutation pi (net->pin), compute M output functions (in output-pin order)
    as 2^N-bit bitvectors over external net patterns t.
    g[j] is an int with bit t = output bit j at that t.
    """
    inv_pi = inverse_perm(pi)
    g = [0] * M

    if verbose:
        print("pi (net->pin):", pi)
        print("inv_pi (pin->net):", inv_pi)
        print("Tracing first", min(show_patterns, 1 << N), "net patterns t:")

    for t in range(1 << N):
        u = net_pattern_to_pin_pattern(t, inv_pi, N)
        z = F[u]  # M-bit int in output-pin order

        # accumulate each output bit into its bitvector position t
        for j in range(M):
            if (z >> j) & 1:
                g[j] |= 1 << t

        if verbose and t < show_patterns:
            print(
                f"  t={t:0{N}b} (nets) -> u={u:0{N}b} (pins) -> F[u]={int_to_bitstr(z, M)} (Z{M}..Z1)"
            )

    if verbose:
        for j in range(M):
            print(f"  Output pin Z{j+1} function bitvector (LSB=t=0): {g[j]:0{1<<N}b}")
    return g

def canonical_signature_for_pi(F: List[int], N: int, M: int, pi: Tuple[int, ...], verbose: bool = False) -> Tuple[int, ...]:
    """
    Output-permutation-invariant signature:
      - compute the M output functions as bitvectors
      - sort them (so any output swap yields same signature)
    """
    g = build_output_functions_bitvectors(F, N, M, pi, verbose=verbose)
    sig = tuple(sorted(g))
    if verbose:
        print("  signature (sorted output bitvectors):", sig)
    return sig

def tuple_repr(pi: Tuple[int, ...], sigma: Tuple[int, ...], N: int, M: int) -> Tuple[str, ...]:
    """
    Represent the full tuple in the user's style:
      (A?, A?, ..., Z?, Z?)
    Where position i corresponds to net(i+1).
    """
    inputs = tuple(f"A{pi[i]+1}" for i in range(N))
    outputs = tuple(f"Z{sigma[j]+1}" for j in range(M))
    return inputs + outputs


# -----------------------------
# Partition into equivalence classes
# -----------------------------

def partition_tuples(F: List[int], N: int, M: int, verbose_one_pi: bool = False) -> Dict[Tuple[int, ...], List[Tuple[int, ...]]]:
    """
    Returns:
      sig2pis: signature -> list of input permutations pi in that class.
    Note:
      output permutations sigma are not listed here because outputs are considered swappable;
      if you want full tuples, you can expand with all sigma permutations afterwards.
    """
    sig2pis: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
    all_pi = list(permutations(range(N)))

    for idx, pi in enumerate(all_pi):
        verbose = (verbose_one_pi and idx == 0)
        sig = canonical_signature_for_pi(F, N, M, pi, verbose=verbose)
        sig2pis.setdefault(sig, []).append(pi)

    return sig2pis

def expand_group_to_full_tuples(pis: List[Tuple[int, ...]], N: int, M: int) -> List[Tuple[str, ...]]:
    """Expand input permutations to full tuples by enumerating all output permutations sigma."""
    all_sigma = list(permutations(range(M)))
    out = []
    for pi in pis:
        for sigma in all_sigma:
            out.append(tuple_repr(pi, sigma, N, M))
    return out


# -----------------------------
# Example (small) for tracing
# -----------------------------
if __name__ == "__main__":
    # Example: N=3 inputs A1,A2,A3; M=2 outputs Z1,Z2
    # Define a toy cell:
    #   Z1 = A1 AND A2
    #   Z2 = A2 XOR A3
    N, M = 3, 2
    F = []
    for u in range(1 << N):
        a1 = (u >> 0) & 1
        a2 = (u >> 1) & 1
        a3 = (u >> 2) & 1
        z1 = a1 & a2
        z2 = a2 ^ a3
        F.append(bits_to_int([z1, z2]))  # [Z1,Z2] in LSB-first

    print("Trace one pi to understand mapping:")
    pi0 = (0, 1, 2)  # net1->A1, net2->A2, net3->A3
    canonical_signature_for_pi(F, N, M, pi0, verbose=True)

    # print("\nPartition all tuples into classes (outputs swappable):")
    # sig2pis = partition_tuples(F, N, M)
    # print("Number of classes K =", len(sig2pis))
    # # print class sizes
    # sizes = sorted((len(pis) for pis in sig2pis.values()), reverse=True)
    # print("Class sizes (in #input-perms):", sizes)

    # # Show one class expanded to full tuples (including all output permutations)
    # one_sig, one_pis = next(iter(sig2pis.items()))
    # full = expand_group_to_full_tuples(one_pis, N, M)
    # print("\nOne class example (first 5 full tuples):")
    # for t in full[:5]:
    #     print(" ", t)
