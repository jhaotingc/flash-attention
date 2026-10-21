# SM90 native FP8 split-KV combine: one warp per output row.
import math
from typing import Optional
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr


class FlashAttentionForwardCombineWarpRow:
    """Split-KV combine for the SM90 native-FP8 forward with at most 32 splits.

    One warp per (query token, head) row and 128 output channels: lane s holds the LSE of
    split s, the LSE max / sum are warp shuffles, and the partial-O channels of every split
    (4 per lane) are loaded into registers together with the LSE, so the combine takes one
    memory round trip and needs no shared memory or CTA-wide barrier. Splits without KV
    blocks have LSE = -inf (and a zero or unwritten partial O); they get weight 0 and are
    skipped by a warp-uniform branch. Launched as a programmatic dependent of the attention
    kernel when use_pdl (griddepcontrol.wait before reading the partials)."""

    def __init__(self, dtype, head_dim: int, max_splits: int, use_pdl: bool = False):
        assert 1 <= max_splits <= 32 and head_dim % 4 == 0
        self.dtype = dtype
        self.head_dim = head_dim
        self.max_splits = max_splits
        self.use_pdl = use_pdl
        self.rows_per_cta = 8
        self.num_threads = cute.arch.WARP_SIZE * self.rows_per_cta

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        stream: cuda.CUstream = None,
    ):
        """mO_partial (splits, total_q, heads, head_dim) fp32, mLSE_partial (splits, total_q,
        heads), mO (total_q, heads, head_dim), mLSE (total_q, heads) or None."""
        grid = (
            cute.ceil_div(mO.shape[0] * mO.shape[1], self.rows_per_cta),
            (self.head_dim + 127) // 128,
            1,
        )
        launch_kwargs = {"use_pdl": True} if const_expr(self.use_pdl) else {}
        self.kernel(mO_partial, mLSE_partial, mO, mLSE).launch(
            grid=grid, block=[self.num_threads, 1, 1], stream=stream, **launch_kwargs
        )

    @cute.kernel
    def kernel(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tidx // cute.arch.WARP_SIZE)
        lane = cute.arch.lane_idx()
        num_heads = mO.shape[1]
        num_splits = mO_partial.shape[0]
        row = bidx * self.rows_per_cta + warp
        q_idx = row // num_heads
        h_idx = row - q_idx * num_heads
        ch_blk = bidy * 32 + lane
        ch_valid = 4 * ch_blk < self.head_dim
        # The partials are written by the preceding (attention) grid.
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        if row < mO.shape[0] * num_heads:
            o_part = cute.make_rmem_tensor((4, self.max_splits), Float32)
            o_part.fill(0.0)
            for s in cutlass.range_constexpr(self.max_splits):
                if s < num_splits and ch_valid:
                    cute.autovec_copy(
                        cute.local_tile(mO_partial[s, q_idx, h_idx, None], (4,), (ch_blk,)),
                        o_part[None, s],
                    )
            # Lane s holds split s's LSE; weights are softmax(LSE) over the splits.
            lse_s = -Float32.inf
            if lane < num_splits:
                lse_s = Float32(mLSE_partial[lane, q_idx, h_idx])
            lse_max = cute.arch.warp_reduction_max(lse_s)
            lse_max_cur = Float32(0.0) if lse_max == -Float32.inf else lse_max
            LOG2_E = math.log2(math.e)
            weight = cute.math.exp2((lse_s - lse_max_cur) * LOG2_E, fastmath=True)
            weight_sum = cute.arch.warp_reduction_sum(weight)
            inv_sum = (
                Float32(0.0) if weight_sum == 0.0 or weight_sum != weight_sum else 1.0 / weight_sum
            )
            weight = weight * inv_sum
            acc = cute.make_rmem_tensor(4, Float32)
            acc.fill(0.0)
            for s in cutlass.range_constexpr(self.max_splits):
                w_s = cute.arch.shuffle_sync(weight, s)
                if w_s > 0.0:  # warp-uniform; skips splits without KV (and their unwritten O)
                    for e in cutlass.range_constexpr(4):
                        acc[e] = acc[e] + w_s * o_part[e, s]
            if ch_valid:
                rO = cute.make_rmem_tensor(4, self.dtype)
                rO.store(acc.load().to(self.dtype))
                cute.autovec_copy(rO, cute.local_tile(mO[q_idx, h_idx, None], (4,), (ch_blk,)))
            if const_expr(mLSE is not None):
                if lane == 0 and bidy == 0:
                    mLSE[q_idx, h_idx] = cute.math.log(weight_sum, fastmath=True) + lse_max
