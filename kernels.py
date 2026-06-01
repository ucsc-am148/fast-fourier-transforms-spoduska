"""STUDENT FILE: implement the Triton kernels and pipeline drivers.

You implement:
  - Six @triton.jit kernels: f1_kernel, f2_kernel, transpose_kernel,
    f4_kernel_L2, dft_kernel, bailey_scale_kernel.
  - The f1_launch and f2_launch grid-choice wrappers around them.
  - The pipeline drivers: f3_launch, f5_launch, _f6_rec, _f7_rec.
  - f6_factor: the chunk-recipe for F6/F7.

You do NOT implement (left given below):
  - The thin launch wrappers _transpose, _fft_chunk, _scale, _lookup_tw.
    These are mechanical "pick the grid and launch one kernel" helpers.
  - The tuning constants F4_L2_BLOCK_B, DFT_BLOCK_B, SCALE_BLOCK,
    TRANSPOSE_BLOCK.

The signatures below are the ones the harness calls -- your job is to fill
the bodies. When your code passes sanity_check.py, you're done.
"""

import math

import torch
import triton
import triton.language as tl


# Tunings -- GIVEN.
F4_L2_BLOCK_B = 2
DFT_BLOCK_B = 16
SCALE_BLOCK = 32
TRANSPOSE_BLOCK = 32


# =============================================================================
# Device-function helper: complex matmul
# =============================================================================
# Implement this once -- f1_kernel, f4_kernel_L2, and dft_kernel all call it.


@triton.jit
def _cdot(a_re, a_im, b_re, b_im):
    """Complex matmul Y = A @ B as four real tl.dot calls.

    Returns (y_re, y_im) in fp32 (out_dtype=tl.float32). Caller is responsible
    for any fp16 down-cast on store. Works at any matmul shape tl.dot accepts.

    Used by f1_kernel, f4_kernel_L2, and dft_kernel. Don't reimplement the
    four-tl.dot expansion at each call site -- implement once here, call
    everywhere.
    """

    # Complex matmul (a_re + i*a_im) @ (b_re + i*b_im) = four real tl.dot calls.
    # real part: a_re*b_re - a_im*b_im ; imag part: a_re*b_im + a_im*b_re
    # out_dtype=tl.float32 keeps accumulation in fp32 (tcFFT contract)
    y_re = (tl.dot(a_re, b_re, out_dtype=tl.float32)
            - tl.dot(a_im, b_im, out_dtype=tl.float32))
    y_im = (tl.dot(a_re, b_im, out_dtype=tl.float32)
            + tl.dot(a_im, b_re, out_dtype=tl.float32))
    return y_re, y_im


# =============================================================================
# Chunk factorization for F6 / F7
# =============================================================================

def f6_factor(N: int) -> list[int]:
    """Factor N = 2^k into FFT chunks.

    Recipe: prefer 256-length chunks (radix-256, handled by f4_kernel_L2), then
    16-length (handled by dft_kernel via the padded radix-16 path), then a
    small leftover in {2, 4, 8} for the remaining bits. chunks[0] is the
    innermost (fastest) input axis. Examples:
        256 -> [256]                4096 -> [256, 16]
        65536 -> [256, 256]         1048576 -> [256, 256, 16]
        64 -> [16, 4]               2 -> [2]
    """
    
    chunks = []
    rem = N
    while rem % 256 == 0:           # prefer radix-256 (F4) chunks
        chunks.append(256)
        rem //= 256
    while rem % 16 == 0 and rem >= 16:   # then radix-16 (padded DFT)
        chunks.append(16)
        rem //= 16
    if rem > 1:                      # small leftover in {2, 4, 8}
        chunks.append(rem)
    return chunks


f7_factor = f6_factor   # F7 reuses F6's chunk recipe


# =============================================================================
# F1: DFT as one dense complex matmul (four tl.dot)
# =============================================================================

@triton.jit
def f1_kernel(
    x_re_ptr, x_im_ptr,    # (B, N) fp16
    W_re_ptr, W_im_ptr,    # (N, N) fp16; W[n, k]
    y_re_ptr, y_im_ptr,    # (B, N) fp32
    B,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Y = X @ W^T as four (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N) tl.dot calls.

    Y[b, n] = sum_k X[b, k] * W[n, k]. Load W in transposed access
    (W_T[k, n] = W[n, k]) so tl.dot reads it the way it wants.

    Use `_cdot(x_re, x_im, W_T_re, W_T_im)` for the per-block complex matmul;
    accumulate its fp32 output into `acc_re` / `acc_im`.

    Dtype contract (same as F4): loads are fp16, `tl.dot` runs with
    `out_dtype=tl.float32` (handled by `_cdot`), accumulator is fp32, store
    is fp32. Allocations in `f1_alloc` already match this -- x_re/x_im are
    fp16, y_re/y_im are fp32.
    """

    # Each program computes one (BLOCK_M, BLOCK_N) tile of the output, looping
    # over K in BLOCK_K chunks.

    # One program computes a (BLOCK_M, BLOCK_M) output tile of Y[b, n].
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # fp32 accumulators (tcFFT contract: dot in fp32, store fp32).
    acc_re = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, N, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # X tile (BLOCK_M, BLOCK_K): row-major X[m, k] lives at m*N + k.
        x_off = offs_m[:, None] * N + offs_k[None, :]
        x_mask = (offs_m[:, None] < B) & (offs_k[None, :] < N)
        x_re = tl.load(x_re_ptr + x_off, mask=x_mask, other=0.0)
        x_im = tl.load(x_im_ptr + x_off, mask=x_mask, other=0.0)
        # Want W^T[k, n] = W[n, k]; W is row-major so W[n, k] is at n*N + k
        w_off = offs_n[None, :] * N + offs_k[:, None]
        w_mask = (offs_k[:, None] < N) & (offs_n[None, :] < N)
        w_re = tl.load(W_re_ptr + w_off, mask=w_mask, other=0.0)
        w_im = tl.load(W_im_ptr + w_off, mask=w_mask, other=0.0)

        p_re, p_im = _cdot(x_re, x_im, w_re, w_im)
        acc_re += p_re
        acc_im += p_im

    # Store the fp32 result tile.
    y_off = offs_m[:, None] * N + offs_n[None, :]
    y_mask = (offs_m[:, None] < B) & (offs_n[None, :] < N)
    tl.store(y_re_ptr + y_off, acc_re, mask=y_mask)
    tl.store(y_im_ptr + y_off, acc_im, mask=y_mask)



def f1_launch(x_re, x_im, W_re, W_im, y_re, y_im):
    """Grid: (cdiv(B, BLOCK_M), cdiv(N, BLOCK_N)). One program tiles a
    (BLOCK_M, BLOCK_N) output square. tl.dot needs all three dims >=16, so B
    should be >= 16.
    """
    
    # Grid: one program per (BLOCK_M, BLOCK_N) output tile
    B, N = x_re.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(N, BLOCK_N))
    f1_kernel[grid](
        x_re, x_im, W_re, W_im, y_re, y_im, B,
        N=N, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )


# =============================================================================
# F2: radix-2 Cooley-Tukey, single program per signal
# =============================================================================
# F3 reuses this kernel! For F2, only BAILEY_EPILOGUE=False, STRIDED_STORE=False need to be implemented.
#
# Call-site cheatsheet:
#   F2 vanilla:  pid -> one signal in (B, N). Grid: (B,).
#                BAILEY_EPILOGUE=False, STRIDED_STORE=False.
#                OUTER_DIM and N_TOTAL unused (pass 1 / 0).
#                bt_*_ptr: pass tw_*_ptr again (sentinel; never read).
#   F2-A (F3):   pid -> (b, n1). Grid: (B*N1,). FFT length N=N2.
#                BAILEY_EPILOGUE=True, STRIDED_STORE=False.
#                OUTER_DIM=N1 (n1 = pid % N1).
#                bt_*_ptr: real Bailey twiddles shape (N1, N2).
#   F2-B (F3):   pid -> (b, k2). Grid: (B*N2,). FFT length N=N1.
#                BAILEY_EPILOGUE=False, STRIDED_STORE=True.
#                OUTER_DIM=N2, N_TOTAL=N1*N2.
#                bt_*_ptr: sentinel.

@triton.jit
def f2_kernel(
    x_re_ptr, x_im_ptr,        # (B, N) fp32 input
    y_re_ptr, y_im_ptr,        # (B, N) fp32 output (layout depends on STRIDED_STORE)
    tw_re_ptr, tw_im_ptr,      # (N/2,) fp32 radix-2 twiddles
    perm_ptr,                   # (N,) int32 bit-reversal index
    bt_re_ptr, bt_im_ptr,       # (OUTER_DIM, N) fp32 Bailey twiddles (BAILEY_EPILOGUE only)
    OUTER_DIM, N_TOTAL,
    N: tl.constexpr,
    LOG2_N: tl.constexpr,
    BAILEY_EPILOGUE: tl.constexpr,
    STRIDED_STORE: tl.constexpr,
):
    """Radix-2 Cooley-Tukey FFT in registers, with optional Bailey epilogue and
    strided store. log2(N) butterfly stages via tl.gather for partner shuffle.
    """

    # ---- VANILLA F2 path (BAILEY_EPILOGUE=False, STRIDED_STORE=False) ----
    pid = tl.program_id(0)              # one program = one length-N signal
    j = tl.arange(0, N)                 # position vector, held in registers

    # Bit-reversed load: v[j] = x[pid, perm[j]].
    perm = tl.load(perm_ptr + j)
    src = pid * N + perm
    vr = tl.load(x_re_ptr + src)
    vi = tl.load(x_im_ptr + src)

    # log2(N) buttergly stages, all in registers (no HBM round-trip).
    for s in tl.static_range(LOG2_N):
        step = 1 << s
        partner = j ^ step
        is_high = (j >> s) & 1  # 1 if j is the high element of its pair

        # partner values via gather
        pr = tl.gather(vr, partner, axis=0)
        pi = tl.gather(vi, partner, axis=0)

        # twiddles: depends only on j's low s bits (same for both pair members)
        ti = (j & (step - 1)) * ( N >> (s + 1))
        wr = tl.load(tw_re_ptr + ti)
        wi = tl.load(tw_im_ptr + ti)

        # w * partner and w * self (complex multiply)
        wpr = wr * pr - wi * pi
        wpi = wr * pi + wi * pr
        wsr = wr * vr - wi * vi
        wsi = wr * vi + wi * vr

        # low: self + w*partner; high: partner - w*self
        cond = is_high == 1
        vr = tl.where(cond, pr - wsr, vr + wpr)
        vi = tl.where(cond, pi - wsi, vi + wpi)

    # # Vanilla store: row-major y[pid, j]
    # dst = pid * N + j
    # tl.store(y_re_ptr + dst, vr)
    # tl.store(y_im_ptr + dst, vi)    ### block replaced with below
    
    # F2-A epilogue: multiply FFT output by Bailey cross-twiddle bt[n1, k].
    if BAILEY_EPILOGUE:
        n1 = pid % OUTER_DIM                # pid encodes (b, n1)
        bt_off = n1 * N + j                 # bt shape (OUTER_DIM, N); k = j
        btr = tl.load(bt_re_ptr + bt_off)
        bti = tl.load(bt_im_ptr + bt_off)
        nr = vr * btr - vi * bti
        ni = vr * bti + vi * btr
        vr, vi = nr, ni
    
    # Store. Vanilla / F2-A: row-major. F2-B: strided by N2 (absorbs T3).
    if STRIDED_STORE:
        b = pid // OUTER_DIM        # pid encodes (b, k2), OUTER_DIM = N2
        k2 = pid % OUTER_DIM
        dst = b * N_TOTAL + j * OUTER_DIM + k2
    else:
        dst = pid * N + j
    
    tl.store(y_re_ptr + dst, vr)
    tl.store(y_im_ptr + dst, vi)


def f2_launch(x_re, x_im, y_re, y_im, tw_re, tw_im, perm):
    """Grid: (B,). One program per length-N signal. Vanilla mode.
    """
    B, N = x_re.shape
    LOG2_N = int(math.log2(N))
    grid = (B,)
    f2_kernel[grid](
        x_re, x_im, y_re, y_im, tw_re, tw_im, perm,
        tw_re, tw_im,          # bt_* sentinel (never read in vanilla)
        1, 0,                  # OUTER_DIM, N_TOTAL unused in vanilla
        N=N, LOG2_N=LOG2_N,
        BAILEY_EPILOGUE=False, STRIDED_STORE=False,
    )



# =============================================================================
# transpose_kernel: (B, R, C) -> (B, C, R), paired re/im
# =============================================================================

@triton.jit
def transpose_kernel(
    x_re_ptr, x_im_ptr,     # (B*R*C,) fp16 or fp32 input
    y_re_ptr, y_im_ptr,     # (B*R*C,) fp16 or fp32 output
    R, C,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Logical (B, R, C) -> (B, C, R) transpose. Grid: (cdiv(R, BLOCK_R),
    cdiv(C, BLOCK_C), B). Each program copies a (BLOCK_R, BLOCK_C) tile.
    """

    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    b = tl.program_id(2)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_r[:, None] < R) & (offs_c[None, :] < C)

    # Input (B, R, C): element [b, r, c] at b*R*C + r*C + c.
    in_off = b * R * C + offs_r[:, None] * C + offs_c[None, :]
    xr = tl.load(x_re_ptr + in_off, mask=mask, other=0.0)
    xi = tl.load(x_im_ptr + in_off, mask=mask, other=0.0)

    # Output (B, C, R): element [b, c, r] at b*C*R + c*R + r.
    out_off = b * C * R + offs_c[None, :] * R + offs_r[:, None]
    tl.store(y_re_ptr + out_off, xr, mask=mask)
    tl.store(y_im_ptr + out_off, xi, mask=mask)


# =============================================================================
# F4: tcFFT radix-16 single-program FFT (N = 256, L = 2)
# =============================================================================
# See the kernel docstring for the tl.permute tuple-literal gotcha.

@triton.jit
def f4_kernel_L2(
    x_re_ptr, x_im_ptr,    # (B, 256) fp16
    y_re_ptr, y_im_ptr,    # (B, 256) or (B//M, 256, M) fp16
    F_re_ptr, F_im_ptr,    # (16, 16) fp16 -- F_16 DFT matrix
    tw_re_ptr, tw_im_ptr,  # (L=2, 16, 16) fp16 stacked stage twiddles
    B, M,
    BLOCK_B: tl.constexpr,
    STAGE_STOP: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """tcFFT length-256 FFT as two stages of (permute + per-stage twiddle +
    length-16 DFT via four tl.dot). fp16 storage, fp32 matmul accumulators.

    `STAGE_STOP` and `M` are both degenerate in vanilla F4 (`STAGE_STOP=L=2`,
    `M=1`). They exist so the same kernel handles two extra uses:
      - `STAGE_STOP=1`: stop after the s=0 stage, for the sanity_check.py
        stage-1 isolation test (no twiddles, no second matmul).
      - `M>1` with `STORE_T=True`: F7's fused FFT-m_0+T3, writing the
        transposed (rows_outer, 256, M) layout the next level expects.

    STORE_T=False (M=1): natural (B, 256) row-major output.
    STORE_T=True  (M>1): transposed (B//M, 256, M) output for F7 fusion.

    Each stage's four-`tl.dot` is one `_cdot` call; cast its fp32 output to
    fp16 before the next stage.

    Dtype contract:
        Loads:           fp16
        Reshape/permute: fp16 (free)
        tl.dot inputs:   fp16, out_dtype=tl.float32  (use _cdot)
        Twiddle mul:     fp32 * fp16 -> fp32
        Inter-stage:     .to(tl.float16) before next iter's reshape
        Store:           fp16
    Forgetting the inter-stage cast doubles register pressure and passes the
    L=2 tolerance, but fails as soon as F6 stacks more stages.

    Triton 3.6 gotcha -- tl.permute requires LITERAL tuples:
        tl.permute(x, (1, 0, 2))                  # works
        perm = (1, 0, 2); tl.permute(x, perm)     # fails
    Inline each stage's permute tuple at the call site; don't store the
    schedule in a loop variable.
    """

    pid = tl.program_id(0)
    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)        # (BLOCK_B,)
    r = tl.arange(0, 16)
    bmask = offs_b[:, None, None] < B

    # Load input tile (BLOCK_B, 16, 16): tile[b, d0, d1] = x[b, d0*16 + d1].
    x_off = offs_b[:, None, None] * 256 + r[None, :, None] * 16 + r[None, None, :]
    tre = tl.load(x_re_ptr + x_off, mask=bmask, other=0.0)
    tim = tl.load(x_im_ptr + x_off, mask=bmask, other=0.0)

    # F_16 DFT matrix (16,16), broadcast across the batch for the batched tl.dot.
    f_off = r[:, None] * 16 + r[None, :]
    fr = tl.load(F_re_ptr + f_off)
    fi = tl.load(F_im_ptr + f_off)
    fr_b = tl.broadcast_to(fr[None, :, :], (BLOCK_B, 16, 16))
    fi_b = tl.broadcast_to(fi[None, :, :], (BLOCK_B, 16, 16))

    # ---- Stage 0: permute identity, no twiddle, length-16 DFT on axis 0 ----
    ore, oim = _cdot(fr_b, fi_b, tre, tim)               # F @ tile, fp32
    tre = ore.to(tl.float16)                             # inter-stage cast (MANDATORY)
    tim = oim.to(tl.float16)

    if STAGE_STOP == 1:
        # Isolation test: stop after stage 0, store tile (b, e1, d1) row-major.
        y_off = offs_b[:, None, None] * 256 + r[None, :, None] * 16 + r[None, None, :]
        tl.store(y_re_ptr + y_off, tre, mask=bmask)
        tl.store(y_im_ptr + y_off, tim, mask=bmask)
        return

    # ---- Stage 1: bring d1 to front, per-stage twiddle, length-16 DFT ----
    tre = tl.permute(tre, (0, 2, 1))                     # literal tuple (Triton 3.6 gotcha)
    tim = tl.permute(tim, (0, 2, 1))

    tw_off = 256 + r[:, None] * 16 + r[None, :]          # stage-1 slice tw[1, m, c]
    twr = tl.load(tw_re_ptr + tw_off).to(tl.float32)
    twi = tl.load(tw_im_ptr + tw_off).to(tl.float32)
    trf = tre.to(tl.float32)
    tif = tim.to(tl.float32)
    mre = trf * twr[None, :, :] - tif * twi[None, :, :]  # fp32 twiddle multiply
    mim = trf * twi[None, :, :] + tif * twr[None, :, :]
    tre = mre.to(tl.float16)                             # back to fp16 for tl.dot
    tim = mim.to(tl.float16)

    ore, oim = _cdot(fr_b, fi_b, tre, tim)               # F @ tile, fp32
    tre = ore.to(tl.float16)
    tim = oim.to(tl.float16)

    # ---- Final store: tile (b, e0, e1), natural freq index n = e0*16 + e1 ----
    n_idx = r[None, :, None] * 16 + r[None, None, :]      # (1,16,16) -> n
    if STORE_T:
        # Fused FFT-m0 + T3: transposed (rows//M, 256, M). b = outer*M + m_idx.
        outer = offs_b // M
        m_idx = offs_b % M
        y_off = outer[:, None, None] * 256 * M + n_idx * M + m_idx[:, None, None]
    else:
        y_off = offs_b[:, None, None] * 256 + n_idx
    tl.store(y_re_ptr + y_off, tre, mask=bmask)
    tl.store(y_im_ptr + y_off, tim, mask=bmask)


# =============================================================================
# dft_kernel: padded length-R DFT for the small chunks (R in {2, 4, 8, 16})
# =============================================================================

@triton.jit
def dft_kernel(
    x_re_ptr, x_im_ptr,     # (rows, R) fp16
    y_re_ptr, y_im_ptr,     # (rows, R) or (rows//M, R, M) fp16
    M_re_ptr, M_im_ptr,     # (16, 16) fp16 padded-R DFT matrix
    rows, M,
    R: tl.constexpr,
    BLOCK_B: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Padded length-R DFT via a (16, 16) tl.dot. STORE_T toggles natural
    vs transposed output (same pattern as f4_kernel_L2).

    One `_cdot(x_re, x_im, MT_re, MT_im)` call replaces the four `tl.dot`
    expansions; cast its fp32 result to fp16 on store.
    """

    pid = tl.program_id(0)
    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)    # (BLOCK_B,) rows
    r16 = tl.arange(0, 16)

    # Load x (rows, R) zero-padded out to 16 columns (cols >= R masked to 0).
    in_mask = (offs_b[:, None] < rows) & (r16[None, :] < R)
    x_off = offs_b[:, None] * R + r16[None, :]
    xr = tl.load(x_re_ptr + x_off, mask=in_mask, other=0.0)
    xi = tl.load(x_im_ptr + x_off, mask=in_mask, other=0.0)

    # Padded DFT matrix (16,16), transposed for x @ M^T: MT[n,k] = M[k,n].
    mt_off = r16[None, :] * 16 + r16[:, None]         # [n,k] -> k*16 + n
    mtr = tl.load(M_re_ptr + mt_off)
    mti = tl.load(M_im_ptr + mt_off)

    # out[b,k] = sum_n x[b,n] * M[k,n] = (x @ MT)[b,k]
    or_, oi = _cdot(xr, xi, mtr, mti)                 # (BLOCK_B,16) fp32
    or_ = or_.to(tl.float16)
    oi = oi.to(tl.float16)

    out_mask = (offs_b[:, None] < rows) & (r16[None, :] < R)
    if STORE_T:
        # (rows//M, R, M): row = outer*M + m_idx.
        outer = offs_b // M
        m_idx = offs_b % M
        y_off = outer[:, None] * R * M + r16[None, :] * M + m_idx[:, None]
    else:
        y_off = offs_b[:, None] * R + r16[None, :]
    tl.store(y_re_ptr + y_off, or_, mask=out_mask)
    tl.store(y_im_ptr + y_off, oi, mask=out_mask)


# =============================================================================
# bailey_scale_kernel: elementwise w_N^{n1 kM} multiply with optional fused T2
# =============================================================================

@triton.jit
def bailey_scale_kernel(
    x_re_ptr, x_im_ptr,     # (rows*m0*M,) fp16 input (logical (rows, m0, M))
    y_re_ptr, y_im_ptr,     # (rows*m0*M,) fp16 output ((rows, m0, M) or (rows, M, m0))
    tw_re_ptr, tw_im_ptr,   # (m0, M) fp16
    m0, M,
    BLOCK_M0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Elementwise complex multiply by bt[n1, kM] over the (rows, m0, M) view.
    fp32 arithmetic, fp16 result. STORE_T=True fuses with a transpose to
    produce (rows, M, m0).

    Grid: (cdiv(m0, BLOCK_M0), cdiv(M, BLOCK_M), rows).
    """

    pid_m0 = tl.program_id(0)
    pid_M = tl.program_id(1)
    row = tl.program_id(2)

    offs_m0 = pid_m0 * BLOCK_M0 + tl.arange(0, BLOCK_M0)
    offs_M = pid_M * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = (offs_m0[:, None] < m0) & (offs_M[None, :] < M)

    # Input (rows, m0, M): element [row, i0, iM] at row*m0*M + i0*M + iM.
    base = row * m0 * M
    in_off = base + offs_m0[:, None] * M + offs_M[None, :]
    xr = tl.load(x_re_ptr + in_off, mask=mask, other=0.0).to(tl.float32)
    xi = tl.load(x_im_ptr + in_off, mask=mask, other=0.0).to(tl.float32)

    # Twiddle table (m0, M).
    tw_off = offs_m0[:, None] * M + offs_M[None, :]
    twr = tl.load(tw_re_ptr + tw_off, mask=mask, other=0.0).to(tl.float32)
    twi = tl.load(tw_im_ptr + tw_off, mask=mask, other=0.0).to(tl.float32)

    yr = xr * twr - xi * twi
    yi = xr * twi + xi * twr

    # STORE_T=False: same (rows, m0, M). STORE_T=True: transposed (rows, M, m0).
    if STORE_T:
        out_off = row * M * m0 + offs_M[None, :] * m0 + offs_m0[:, None]
    else:
        out_off = base + offs_m0[:, None] * M + offs_M[None, :]
    tl.store(y_re_ptr + out_off, yr.to(tl.float16), mask=mask)
    tl.store(y_im_ptr + out_off, yi.to(tl.float16), mask=mask)


# =============================================================================
# Thin launch wrappers -- GIVEN, do not edit
# =============================================================================

def _transpose(in_re, in_im, out_re, out_im, B, R, C):
    """Logical (B, R, C) -> (B, C, R) transpose, paired re/im."""
    grid = (triton.cdiv(R, TRANSPOSE_BLOCK), triton.cdiv(C, TRANSPOSE_BLOCK), B)
    transpose_kernel[grid](
        in_re, in_im, out_re, out_im, R, C,
        BLOCK_R=TRANSPOSE_BLOCK, BLOCK_C=TRANSPOSE_BLOCK,
    )


def _fft_chunk(in_re, in_im, out_re, out_im, rows, m, plan, M=1, store_t=False):
    """Length-m FFT over `rows` contiguous (rows, m) signals.

    M / store_t control the output layout:
      store_t=False, M=1: natural (rows, m) row-major (F6 leaf path)
      store_t=True,  M>1: transposed (rows//M, m, M) (F7 fused FFT-m0+T3)
    """
    if m == 256:
        f4_plan = plan['f4_plan']
        f4_kernel_L2[(triton.cdiv(rows, F4_L2_BLOCK_B),)](
            in_re.view(rows, 256), in_im.view(rows, 256),
            out_re.view(rows, 256), out_im.view(rows, 256),
            f4_plan['F_re'], f4_plan['F_im'],
            f4_plan['tw_re'], f4_plan['tw_im'],
            rows, M,
            BLOCK_B=F4_L2_BLOCK_B, STAGE_STOP=f4_plan['L'], STORE_T=store_t,
            num_warps=4, num_stages=1,
        )
    else:
        M_re, M_im = plan['dft_mats'][m]
        dft_kernel[(triton.cdiv(rows, DFT_BLOCK_B),)](
            in_re.view(rows, m), in_im.view(rows, m),
            out_re.view(rows, m), out_im.view(rows, m),
            M_re, M_im, rows, M,
            R=m, BLOCK_B=DFT_BLOCK_B, STORE_T=store_t,
        )


def _scale(in_re, in_im, out_re, out_im, rows, m0, M, twr, twi, store_t=False):
    """Bailey scale over logical (rows, m0, M)."""
    grid = (triton.cdiv(m0, SCALE_BLOCK), triton.cdiv(M, SCALE_BLOCK), rows)
    bailey_scale_kernel[grid](
        in_re, in_im, out_re, out_im, twr, twi,
        m0, M, BLOCK_M0=SCALE_BLOCK, BLOCK_M=SCALE_BLOCK, STORE_T=store_t,
    )


def _lookup_tw(plan, m0, M, N_i):
    """Find the precomputed Bailey twiddle table for (m0, M, N_i) in plan['tw']."""
    for (a, b, n, tr, ti) in plan['tw']:
        if a == m0 and b == M and n == N_i:
            return tr, ti
    raise KeyError(f"no twiddle table for (m0={m0}, M={M}, N={N_i})")


# =============================================================================
# F3 pipeline: 4-step Bailey six-step (T1 -> F2-A -> T2 -> F2-B)
# =============================================================================

def f3_launch(in_re, in_im, out_re, out_im, mid_re, mid_im, plan, B):
    """Run the 4-step F3 pipeline. Buffer ping-pong: in -> mid -> out -> mid
    -> out. The Bailey twiddle fuses into F2-A (BAILEY_EPILOGUE=True), and
    the would-be T3 is absorbed by F2-B (STRIDED_STORE=True).

    Steps:
      1. T1 (transpose): x[b, n2, n1] -> A[b, n1, n2]
      2. F2-A:           length-N2 FFT over (B*N1) signals with Bailey epilogue
      3. T2 (transpose): Z[b, n1, k2] -> Z'[b, k2, n1]
      4. F2-B:           length-N1 FFT over (B*N2) signals with strided store
    """

    N1 = plan['N1']
    N2 = plan['N2']

    # Step 1 — T1: (B, N2, N1) -> (B, N1, N2).  in -> mid
    _transpose(in_re, in_im, mid_re, mid_im, B, N2, N1)

    # Step 2 — F2-A: length-N2 FFT over B*N1 signals + Bailey epilogue.  mid -> out
    f2_kernel[(B * N1,)](
        mid_re, mid_im, out_re, out_im,
        plan['tw_re_n2'], plan['tw_im_n2'], plan['perm_n2'],
        plan['bt_re'], plan['bt_im'],
        N1, 0,                                 # OUTER_DIM=N1, N_TOTAL unused
        N=N2, LOG2_N=plan['LOG2_N2'],
        BAILEY_EPILOGUE=True, STRIDED_STORE=False,
    )

    # Step 3 — T2: (B, N1, N2) -> (B, N2, N1).  out -> mid
    _transpose(out_re, out_im, mid_re, mid_im, B, N1, N2)

    # Step 4 — F2-B: length-N1 FFT over B*N2 signals + strided store.  mid -> out
    f2_kernel[(B * N2,)](
        mid_re, mid_im, out_re, out_im,
        plan['tw_re_n1'], plan['tw_im_n1'], plan['perm_n1'],
        plan['tw_re_n1'], plan['tw_im_n1'],    # bt sentinel (never read)
        N2, N1 * N2,                           # OUTER_DIM=N2, N_TOTAL=N
        N=N1, LOG2_N=plan['LOG2_N1'],
        BAILEY_EPILOGUE=False, STRIDED_STORE=True,
    )


# =============================================================================
# F5 pipeline: 6-step Bailey at N1=N2=256 with F4 as inner FFT
# =============================================================================

def f5_launch(in_re, in_im, b0_re, b0_im, b1_re, b1_im, b2_re, b2_im, plan, B):
    """Run the 6-step F5 pipeline at N = 65536 = 256 * 256.

    Buffer ping-pong: in -> b0 -> b1 -> b0 -> b1 -> b2 -> b0 (final).
    The Bailey twiddle is NOT fused into F4 (F4 stays unmodified), so this is
    6 launches; F7 generalizes the fusion idea recursively.

    Steps:
      1. T1:    x[b, n2, n1] -> A[b, n1, n2]
      2. FFT-A: length-256 FFT along last axis -> Y[b, n1, k2]
      3. Scale: Z[b, n1, k2] = Y[b, n1, k2] * bt[n1, k2]
      4. T2:    Z[b, n1, k2] -> Z'[b, k2, n1]
      5. FFT-B: length-256 FFT along last axis -> V[b, k2, k1]
      6. T3:    V[b, k2, k1] -> X[b, k1, k2]   (final in b0)
    """

    N1 = plan['N1']    # 256
    N2 = plan['N2']    # 256

    # 1. T1:    (B, N2, N1) -> (B, N1, N2).          in -> b0
    _transpose(in_re, in_im, b0_re, b0_im, B, N2, N1)
    # 2. FFT-A: length-N2 FFT over B*N1 rows.        b0 -> b1
    _fft_chunk(b0_re, b0_im, b1_re, b1_im, B * N1, N2, plan)
    # 3. Scale: Y[b, n1, k2] *= bt[n1, k2].          b1 -> b0
    _scale(b1_re, b1_im, b0_re, b0_im, B, N1, N2, plan['bt_re'], plan['bt_im'])
    # 4. T2:    (B, N1, N2) -> (B, N2, N1).          b0 -> b1
    _transpose(b0_re, b0_im, b1_re, b1_im, B, N1, N2)
    # 5. FFT-B: length-N1 FFT over B*N2 rows.        b1 -> b2
    _fft_chunk(b1_re, b1_im, b2_re, b2_im, B * N2, N1, plan)
    # 6. T3:    (B, N2, N1) -> (B, N1, N2).          b2 -> b0 (final)
    _transpose(b2_re, b2_im, b0_re, b0_im, B, N2, N1)


# =============================================================================
# F6 / F7 recursion
# =============================================================================
# Per level i with chunks = [m_0, m_1, ..., m_{p-1}], M = prod(chunks[1:]):
#   T1 :       (rows, M, m_0) -> (rows, m_0, M)
#   recurse:   length-M FFT over (rows*m_0, M)
#   Scale :    y *= w_{N_i}^{n_1 k_M}            (n_1 = the m_0 digit)
#   T2 :       (rows, m_0, M) -> (rows, M, m_0)
#   FFT-m_0 :  length-m_0 FFT over (rows*M, m_0)
#   T3 :       (rows, M, m_0) -> (rows, m_0, M)   [F6 only; F7 fuses]

def _f6_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Recursive 2-factor Bailey split. Leaf (len(chunks)==1) is one
    _fft_chunk call; non-leaf is the 6-step pipeline above.

    Returns the (re, im) cycler-managed buffers holding the (rows, prod(chunks))
    FFT result.
    """

    m0 = chunks[0]

    # Leaf: a single length-m0 FFT, natural output.
    if len(chunks) == 1:
        out_re, out_im = cyc.next()
        _fft_chunk(cur_re, cur_im, out_re, out_im, rows, m0, plan)
        return out_re, out_im

    M = math.prod(chunks[1:])
    Ni = m0 * M
    twr, twi = _lookup_tw(plan, m0, M, Ni)

    # T1: (rows, M, m0) -> (rows, m0, M)
    t1_re, t1_im = cyc.next()
    _transpose(cur_re, cur_im, t1_re, t1_im, rows, M, m0)

    # recurse: length-M FFT over (rows*m0, M)
    rec_re, rec_im = _f6_rec(t1_re, t1_im, rows * m0, chunks[1:], plan, cyc)

    # Scale: (rows, m0, M) *= w_{Ni}^{n1 * kM} 
    sc_re, sc_im = cyc.next()
    _scale(rec_re, rec_im, sc_re, sc_im, rows, m0, M, twr, twi)

    # T2: (rows, m0, M) -> (rows, M, m0)
    t2_re, t2_im = cyc.next()
    _transpose(sc_re, sc_im, t2_re, t2_im, rows, m0, M)

    # FFT-m0: length-m0 FFT over (rows*M, m0)
    f_re, f_im = cyc.next()
    _fft_chunk(t2_re, t2_im, f_re, f_im, rows * M, m0, plan)

    # T3: (rows, M, m0) -> (rows, m0, M)
    t3_re, t3_im = cyc.next()
    _transpose(f_re, f_im, t3_re, t3_im, rows, M, m0)

    return t3_re, t3_im


def _f7_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Same recursion as _f6_rec but with Scale+T2 fused (store_t=True on
    bailey_scale_kernel) and FFT-m_0+T3 fused (store_t=True, M=M on the inner
    FFT kernel). Output should be bitwise-equal to _f6_rec.
    """

    m0 = chunks[0]

    # Leaf: identical to F6 -- one length-m0 FFT, natural output.
    if len(chunks) == 1:
        out_re, out_im = cyc.next()
        _fft_chunk(cur_re, cur_im, out_re, out_im, rows, m0, plan)
        return out_re, out_im

    M = math.prod(chunks[1:])
    Ni = m0 * M
    twr, twi = _lookup_tw(plan, m0, M, Ni)

    # T1: (rows, M, m0) -> (rows, m0, M)
    t1_re, t1_im = cyc.next()
    _transpose(cur_re, cur_im, t1_re, t1_im, rows, M, m0)

    # recurse: length-M FFT over (rows*m0, M)
    rec_re, rec_im = _f7_rec(t1_re, t1_im, rows * m0, chunks[1:], plan, cyc)

    # Scale + T2 fused: scale (rows, m0, M), write transposed (rows, M, m0).
    sc_re, sc_im = cyc.next()
    _scale(rec_re, rec_im, sc_re, sc_im, rows, m0, M, twr, twi, store_t=True)

    # FFT-m0 + T3 fused: length-m0 FFT over (rows*M, m0), write transposed (rows, m0, M).
    f_re, f_im = cyc.next()
    _fft_chunk(sc_re, sc_im, f_re, f_im, rows * M, m0, plan, M=M, store_t=True)

    return f_re, f_im