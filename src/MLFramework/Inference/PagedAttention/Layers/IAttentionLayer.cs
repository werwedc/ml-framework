using RitterFramework.Core.Tensor;
using MLFramework.Core;

namespace MlFramework.Inference.PagedAttention.Layers;

/// <summary>
/// Interface for attention layers.
/// Provides a contract for computing attention with various implementations.
/// </summary>
public interface IAttentionLayer
{
    /// <summary>
    /// Compute attention output.
    /// </summary>
    /// <param name="hiddenStates">Input hidden states [batchSize, seqLen, hiddenSize]</param>
    /// <returns>Attention output tensor</returns>
    Tensor ComputeAttention(Tensor hiddenStates);
}
