using RitterFramework.Core.Tensor;
using MLFramework.Core;
using MlFramework.Inference.PagedAttention.Phases;

namespace MlFramework.Inference.PagedAttention.Kernels;

/// <summary>
/// Interface for paged attention computation kernels.
/// Implementations can be optimized for different hardware (CUDA, HIP, CPU).
/// </summary>
public interface IPagedAttentionKernel
{
    /// <summary>
    /// Compute scaled dot-product attention using paged KV cache.
    /// </summary>
    /// <param name="query">Query tensor [batch, numQueries, numHeads, headDim]</param>
    /// <param name="cachedKeys">Cached keys [batch, numCached, numHeads, headDim]</param>
    /// <param name="cachedValues">Cached values [batch, numCached, numHeads, headDim]</param>
    /// <param name="phase">Execution phase (Prefill or Decode)</param>
    /// <param name="scale">Scaling factor for attention scores (default: 1/sqrt(headDim))</param>
    /// <returns>Attention output [batch, numQueries, numHeads, headDim]</returns>
    Tensor ComputePagedAttention(
        Tensor query,
        Tensor cachedKeys,
        Tensor cachedValues,
        AttentionPhase phase,
        float scale = -1.0f
    );

    /// <summary>
    /// Check if this kernel supports the given device.
    /// </summary>
    bool SupportsDevice(Device device);
}
