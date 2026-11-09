# haar_fast_numba.py
from collections import deque
import math
import numpy as np
import numba as nb


# ---------- Utilities ----------
def _prefix_sums_int64(x: np.ndarray) -> np.ndarray:
    """
    Exclusive prefix sums P of x with int64 accumulation.
    P has shape (len(x)+1,), P[0]=0, P[i]=sum(x[:i]).
    """
    P = np.empty(x.size + 1, dtype = np.int64)
    P[0] = 0
    #np.cumsum(x.astype(np.int64, copy=False), out=P[1:])
    np.cumsum(x, dtype=np.int64, out=P[1:])
    return P


@nb.njit(cache=True, fastmath=False)
def _choose_break_numba(
    prefix: np.ndarray,
    s: int,
    e: int,
    p0: float,
    rtol: float,
    atol: float,
) -> int:
    L = e - s
    n_candidates = L - 1
    if n_candidates <= 0:
        return (s + e) // 2

    t0 = int((1.0 - p0) * n_candidates)
    t1 = int(p0 * n_candidates)
    if t1 <= t0:
        t0, t1 = 0, n_candidates

    inv_total = 1.0 / (L + 1.0)  # == 1.0 / (e - s + 1)
    seg_center = (s + e) // 2

    # Initialize with the first candidate in window
    best_b = s + 1 + t0
    prefix_sum = float(prefix[best_b] - prefix[s])
    suffix_sum = float(prefix[e] - prefix[best_b])
    inv_prefix = 1.0 / float(best_b - s + 1)
    inv_suffix = 1.0 / float(e - best_b)
    v_high = math.sqrt(inv_prefix - inv_total)
    v_low = -math.sqrt(inv_suffix - inv_total)
    best_score = prefix_sum * v_low + suffix_sum * v_high
    best_dist = abs(best_b - seg_center)

    for idx in range(t0 + 1, t1):
        b = s + 1 + idx

        prefix_sum = float(prefix[b] - prefix[s])
        suffix_sum = float(prefix[e] - prefix[b])

        inv_prefix = 1.0 / float(b - s + 1)
        inv_suffix = 1.0 / float(e - b)

        v_high = math.sqrt(inv_prefix - inv_total)
        v_low = -math.sqrt(inv_suffix - inv_total)

        score = prefix_sum * v_low + suffix_sum * v_high

        if score > best_score:
            best_score = score
            best_b = b
            best_dist = abs(b - seg_center)
        else:
            # isclose(score, best_score) with symmetric scaling
            # (use current score and best_score only; no second pass needed)
            if abs(score - best_score) <= (atol + rtol * max(abs(score), abs(best_score))):
                dist = abs(b - seg_center)
                if dist < best_dist:
                    best_b = b
                    best_dist = dist

    return best_b



# ---------- Haar coefficients & basis helpers ----------
def _haar_high_low(s: int, b: int, e: int):
    """
    Compute the (high, low) coefficients for (s, b, e).
    Mirrors the original formulas.
    """
    inv_total = 1.0 / (e - s + 1.0)
    high = math.sqrt(1.0 / (b - s + 1.0) - inv_total)
    low = -math.sqrt(1.0 / (e - b) - inv_total)
    return high, low


def create_basis_form_fast(s: int, b: int, e: int):
    """
    Same layout as your original `create_basis_form`:
      [depth, e-s, s, (high, b-s), (low, e-b)]
    (Depth is set by the caller.)
    """
    high, low = _haar_high_low(s, b, e)
    return [0, e - s, s, (high, b - s), (low, e - b)]


# ---------- Public APIs ----------
def choose_break_fast(
    signal: np.ndarray, s: int, e: int, p0: float = 0.95,
    rtol: float = 1e-12, atol: float = 1e-15
) -> int:
    """
    Convenience wrapper if you need a one-off call.
    Builds prefix once here; for bulk calls use generate_haar_basis_fast which reuses prefix.
    """
    if not (0.5 <= p0 < 1.0):
        raise ValueError("p0 must be in [.5, 1).")
    prefix = _prefix_sums_int64(signal)
    return int(_choose_break_numba(prefix, int(s), int(e), float(p0), float(rtol), float(atol)))


def generate_haar_basis_fast(
    signal: np.ndarray,
    p0: float = 0.95,
    length=20,
    rtol: float = 1e-12,
    atol: float = 1e-15,
    return_format: str = "python",  # "python" (default) or "array"
):
    """
    Same as before; if return_format == "array", returns a compact structured array:
      dtype=[('depth','i4'),('span','i4'),('s','i4'),
             ('high','f8'),('len_left','i4'),('low','f8'),('len_right','i4')]
    """
    if callable(length):
        d = int(length(signal.size))
    elif isinstance(length, int):
        d = int(length)
    else:
        raise ValueError("Depth must be an integer or a function.")
    if not (0.5 <= p0 < 1.0):
        raise ValueError("p0 must be in [.5, 1).")

    d = max(d, 2)
    n = int(signal.size)

    # Ensure no accidental copies elsewhere (contiguity)
    signal = np.ascontiguousarray(signal)
    prefix = _prefix_sums_int64(signal)

    # Seed like before
    root_mean = float(signal.mean())
    done = [[0, n, 0, (root_mean, n), (0.0, 0)]]

    todo = deque()
    todo.append((0, 0, n))

    while todo:
        depth, s, e = todo.popleft()
        if e - s >= d:
            b = int(_choose_break_numba(prefix, s, e, float(p0), float(rtol), float(atol)))
            if b <= s or b >= e:
                b = (s + e) // 2  # defensive

            node = create_basis_form_fast(s, b, e)
            node[0] = depth
            done.append(node)

            todo.append((depth + 1, s, b))
            todo.append((depth + 1, b, e))

    filtered = [d0 for d0 in done if (d0[3][1] > 10) and (d0[4][1] > 10)]

    if return_format == "python":
        return [done[0]] + filtered

    # Compact: structured array (reduces Python overhead significantly)
    dt = np.dtype([
        ("depth", "i4"), ("span", "i4"), ("s", "i4"),
        ("high", "f8"), ("len_left", "i4"),
        ("low", "f8"), ("len_right", "i4"),
    ])
    arr = np.empty(len(filtered) + 1, dtype=dt)
    # root entry
    arr[0] = (0, n, 0, root_mean, n, 0.0, 0)
    # children
    for i, node in enumerate(filtered, start=1):
        depth, span, s0, (high, len_left), (low, len_right) = node
        arr[i] = (depth, span, s0, high, len_left, low, len_right)
    return arr


def group_means_fast(x: np.ndarray, n: int) -> np.ndarray:
    """
    Vectorized equivalent of your group_means (linspace-based partition).
    Partitions x into n nearly equal contiguous bins and returns their means.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    idx = np.linspace(0, x.size, n + 1, dtype=int)
    sums = np.add.reduceat(x, idx[:-1])
    counts = np.diff(idx)
    return sums / counts
