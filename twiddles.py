"""STUDENT FILE: implement the canonical twiddle helpers.

Three patterns (so seven inconsistently-named lecture helpers collapse to ~3):
  1. radix-2 length-N/2 twiddles   make_radix2_twiddles
  2. per-stage radix-16 twiddles   make_radix16_twiddles
  3. Bailey cross-term twiddles    make_bailey_cross_twiddles

Plus two scaffolding tables (full DFT, padded-R DFT) and the bit-reversal
permutation. Use the forward-FFT sign convention exp(-2*pi*i * ...) and
return (re, im) tuples of separate real-valued tensors everywhere.

When you implement each function, the signature should match the docstring
exactly -- the harness expects (re, im) tuples with specific shapes/dtypes,
and sanity_check.py will FAIL if you return something else.
"""

import math

import torch


# =============================================================================
# Pattern 1: radix-2 length-N/2 twiddles  (F2, F3)
# =============================================================================

def make_radix2_twiddles(
    N: int,
    dtype: torch.dtype = torch.float32,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    """w_N^k for k in [0, N/2). Returns (tw_re, tw_im), each shape (N//2,).

    Used by the radix-2 butterfly: stage s reads twiddle at index
    (k & (2**s - 1)) * (N >> (s+1)), so the table only needs the lower half
    of one full period."""
    # raise NotImplementedError("TO DO: implement make_radix2_twiddles")

    # k = [0, 1, ... N/2 - 1] in float64
    k = torch.arange(N // 2, device=device, dtype=torch.float64)

    # forward FFT sign convention: negative angle exp(-2*pi*i * k / N)
    ang = -2.0 * math.pi * k / N

    # return cosine and sine of angle in float32
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)

# =============================================================================
# Pattern 2: per-stage radix-16 twiddles  (F4; reused by F5/F6/F7 via F4)
# =============================================================================
# The index bookkeeping for this helper is given -- the per-stage permute
# schedule means the column-axis labels at stage s are a mix of already-
# transformed output digits and not-yet-transformed input digits, in an
# order set by the cumulative permutation history. 

def _column_axis_labeling(L: int) -> list[tuple]:
    """Track axis labels through the per-stage permute schedule.

    Convention: input n decomposes as n = sum_i d_i * 16^(L-1-i) with d_0 the
    high digit; output k similarly with e_i. Initial tile has axis i labeled
    ('d', i). At each stage s the kernel applies perm = (s,) + (others in
    original order), bringing axis s to position 0; the four-tl.dot then
    transforms position 0 from ('d', s) to ('e', L-1-s).

    Returns a list of length L; entry s is the tuple of L-1 labels at axis
    positions 1..L-1 of the (16,)*L tile *after* the stage-s permute.
    """
    A = [('d', i) for i in range(L)]
    out = []
    for s in range(L):
        P = [A[s]] + [A[i] for i in range(L) if i != s]
        out.append(tuple(P[1:]))
        A = [('e', L - 1 - s)] + P[1:]
    return out


def make_radix16_twiddles(
    N: int,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-stage radix-16 Cooley-Tukey twiddles, stacked. Returns
    (tw_re, tw_im), each shape (L, 16, N//16) fp16. L = log_16(N).

    Stage-0 slice is ones (kernel skips the multiply on s == 0). Stage s > 0
    is built from the labeling above via:
        tw[m, c] = exp(-2*pi*i * m * t / 16^(s+1))
        t = sum_{j=0}^{s-1} e_{L-1-j}_value(c) * 16^j
    where e_{L-1-j}_value(c) reads the base-16 digit of c at the position
    given by _column_axis_labeling(L)[s].
    """
    # raise NotImplementedError("TO DO: implement make_radix16_twiddles")

	# = 16^L, so there are L radix-16 stages. Each stage gets its own
    # (16, N//16) twiddle matrix; we stack them into (L, 16, N//16)
    L = int (round(math.log(N, 16)))
    cols = N // 16  # = 16^(L-1); the column index space

    # labeling[s] tells us, AFTER stage s's permute, which digit ('d'=input,
    # 'e'=output) sits at each of the L-1 column-axis positions (1..L-1).
    labels = _column_axis_labeling(L)

    # m = row index 0..15 (the digit transformed at this stage), as a column
    # vector so it broadcasts against the row of column indices c.
    m = torch.arange(16, device=device, dtype=torch.float64).reshape(16, 1)
    c = torch.arange(cols, device=device, dtype=torch.int64)    # 0..N/16 - 1

    re_stages, im_stages = [], []
    for s in range(L):
        if s == 0:
            # Stage 0 has no earlier output digits to mix in, so t=0 and the 
            # twiddle is identically 1. The kernel skips the multiply here;
            # we still emit a ones-slice to keep the (L, ...) shape uniform.
            re = torch.ones(16, cols, device=device, dtype=torch.float64)
            im = torch.zeros(16, cols, device=device, dtype=torch.float64)
        else:
            lab = labels[s]     # lab[i] = label at column-axis position i+1
            #  Build t(c) = sum_{j<s} (digit e_{L-1-j} carried by c) * 16^j.
            t = torch.zeros(cols, device=device, dtype=torch.int64)
            for j in range(s):
                target = ('e', L - 1 - j)   # output digit we need
                # find which column-axis position currently holds that digit
                p = next(i + 1 for i, l in enumerate(lab) if l == target)
                # extract that base-16 digit out of the flattened column index c
                # (row-major: position 1 is most significant -> place 16^(L-1-p))
                digit = (c // (16 ** (L - 1 - p))) % 16
                t = t + digit ** (16 ** j)
            # tw[m, c] = exp(-2*pi*i * m * t / 16^(s+1)). Negative angle =
            # forward-FFT convention. float64 math, cast to fp16 at the end.
            ang = (-2.0 * math.pi * m * t.reshape(1, cols).to(torch.float64)
                   / (16 ** (s + 1)))
            re = torch.cos(ang)
            im = torch.sin(ang)
        re_stages.append(re)
        im_stages.append(im)
    
    # stack per-stage matrices -> (L, 16, N//16), cast to the tcFFT fp16 storage.
    tw_re = torch.stack(re_stages).to(torch.float16)
    tw_im = torch.stack(im_stages).to(torch.float16)
    return tw_re, tw_im


# =============================================================================
# Pattern 3: Bailey cross-term twiddles  (F3, F5, F6, F7)
# =============================================================================

def make_bailey_cross_twiddles(
    m0: int,
    M: int,
    N: int,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    """w_N^{n1 * kM} for n1 in [0, m0), kM in [0, M). Returns (re, im), each
    shape (m0, M).

    F3 calls this with dtype=torch.float32 (the radix-2 tier is fp32);
    F5/F6/F7 call it with dtype=torch.float16 (the tcFFT tier is fp16). The
    Bailey identity holds for any N >= m0 * M; in practice N == m0 * M.
    """
    # raise NotImplementedError("TO DO: implement make_bailey_cross_twiddles")

    # bt[n1, kM] = exp(-2*pi*i * n1 * kM / N). n1 indexes rows (0..m0-1),
    # kM indexes columns (0..M-1); outer product gives every n1*kM product.
    n1 = torch.arange(m0, device=device, dtype=torch.float64).reshape(m0, 1)
    kM = torch.arange(M, device=device, dtype=torch.float64).reshape(1, M)
    ang = -2.0 * math.pi * n1 * kM / N
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)


# =============================================================================
# Scaffolding tables
# =============================================================================

def make_dft_matrix(
    N: int,
    dtype: torch.dtype = torch.float16,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full (N, N) DFT matrix. Returns (W_re, W_im).

    W[j, k] = exp(-2*pi*i * j * k / N). Used by F1 (DFT-as-complex-matmul).
    """
    # raise NotImplementedError("TO DO: implement make_dft_matrix")

    # W[j, k] = exp(-2*pi*i * j * k / N). Outer product of row index j and
    # column index k gives every j*k product; same cos/sin split as radix-2.
    j = torch.arange(N, device=device, dtype=torch.float64).reshape(N, 1)
    k = torch.arange(N, device=device, dtype=torch.float64).reshape(1, N)
    ang = -2.0 * math.pi * j * k / N
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)


def make_dft_R_padded(
    R: int,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Length-R DFT padded to (16, 16) fp16. Returns (M_re, M_im).

    Pad the length-R row to 16 with zeros, hit it with a (16, 16) matrix whose
    first R columns are F_R (rows wrap mod R), take the first R output rows.
    This makes the >=16x16 tl.dot requirement hold for all R in {2, 4, 8, 16}.
    """
    # raise NotImplementedError("TO DO: implement make_dft_R_padded")

    # A 16x16 matrix where top-left R x R corner is the length-R DFT F_R, with
    # rows/cols wrapping mod R. The kernel zero-pads the input past column R, so
    # the wrapped entries never actually contribute -- padding just satisfies
    # tl.dot's 16x16 minimum shape.
    k = torch.arange(16, device=device, dtype=torch.int64).reshape(16, 1)
    n = torch.arange(16, device=device, dtype=torch.int64).reshape(1, 16)
    ang = -2.0 * math.pi * ((k % R) * (n % R)).to(torch.float64) / R
    return torch.cos(ang).to(torch.float16), torch.sin(ang).to(torch.float16)


def bit_reversal_perm(N: int, device: str = 'cuda') -> torch.Tensor:
    """Length-N bit-reversal permutation as a (N,) int32 tensor.

    rev[i] is the integer whose n_bits=log2(N) binary representation is i's
    bits in reversed order.
    """
    # raise NotImplementedError("TO DO: implement bit_reversal_perm")

    num_bits = N.bit_length() - 1

	# indices [0, N-1]
    indices = torch.arange(N, dtype=torch.int32, device=device)

	# rev tensor
    reversed_indices = torch.zeros(N, dtype=torch.int32, device=device)

    # find target bit, extract, move to slot 1, isolate with & 1, then place at
    # target bit. Accumulate with bitwise OR
    for source_bit in range(num_bits):
        target_bit = num_bits - 1 - source_bit
        bit_value = (indices >> source_bit) & 1
        shifted_bit = bit_value << target_bit
        reversed_indices = reversed_indices | shifted_bit

    return reversed_indices





