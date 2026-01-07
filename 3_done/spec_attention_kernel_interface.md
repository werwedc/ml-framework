# Spec: Paged Attention Kernel Interface

## Overview
Define the interface and a reference implementation for paged attention computation kernels. These kernels handle the actual attention computation using gathered KV tensors from non-contiguous memory blocks.

## Target Directory
`src/MlFramework/Inference/PagedAttention/Kernels/`

## Classes to Implement

### IPagedAttentionKernel (Interface)
```csharp
using MlFramework.Tensor;

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
```

### StandardPagedAttentionKernel (Reference Implementation)
```csharp
namespace MlFramework.Inference.PagedAttention.Kernels;

/// <summary>
/// Reference implementation of paged attention using standard tensor operations.
/// This implementation is not optimized but provides correct results for testing.
/// For production, use optimized CUDA/HIP implementations.
/// </summary>
public class StandardPagedAttentionKernel : IPagedAttentionKernel
{
    private readonly float _scale;

    public StandardPagedAttentionKernel(int headDim)
    {
        _scale = 1.0f / MathF.Sqrt(headDim);
    }

    public Tensor ComputePagedAttention(
        Tensor query,
        Tensor cachedKeys,
        Tensor cachedValues,
        AttentionPhase phase,
        float scale = -1.0f)
    {
        var actualScale = scale > 0 ? scale : _scale;

        // Input shapes:
        // query:      [batch, numQueries, numHeads, headDim]
        // cachedKeys: [batch, numCached, numHeads, headDim]
        // cachedValues: [batch, numCached, numHeads, headDim]

        int batchSize = query.Shape[0];
        int numQueries = query.Shape[1];
        int numHeads = query.Shape[2];
        int headDim = query.Shape[3];
        int numCached = cachedKeys.Shape[1];

        // Compute attention scores: Q * K^T
        // Result shape: [batch, numQueries, numHeads, numCached]
        var attentionScores = MatmulQueryKey(
            query,
            cachedKeys,
            batchSize,
            numQueries,
            numHeads,
            headDim,
            numCached
        );

        // Scale scores
        attentionScores = attentionScores * actualScale;

        // Create causal mask for decode phase
        if (phase == AttentionPhase.Decode && numCached > numQueries)
        {
            attentionScores = ApplyCausalMask(
                attentionScores,
                batchSize,
                numQueries,
                numHeads,
                numCached
            );
        }

        // Compute attention weights using softmax
        var attentionWeights = Softmax(attentionScores, dim: -1);

        // Compute output: weights * V
        // Result shape: [batch, numQueries, numHeads, headDim]
        var output = MatmulWeightValue(
            attentionWeights,
            cachedValues,
            batchSize,
            numQueries,
            numHeads,
            numCached,
            headDim
        );

        return output;
    }

    private Tensor MatmulQueryKey(
        Tensor query,
        Tensor keys,
        int batchSize,
        int numQueries,
        int numHeads,
        int headDim,
        int numCached)
    {
        // Query: [batch, numQueries, numHeads, headDim]
        // Keys:  [batch, numCached, numHeads, headDim]
        // Output: [batch, numQueries, numHeads, numCached]

        // Reshape for batched matmul
        // Query -> [batch * numHeads, numQueries, headDim]
        var queryReshaped = query.Reshape(
            batchSize * numHeads,
            numQueries,
            headDim
        );

        // Keys -> [batch * numHeads, numCached, headDim]
        var keysReshaped = keys.Reshape(
            batchSize * numHeads,
            numCached,
            headDim
        );

        // Transpose keys: [batch * numHeads, headDim, numCached]
        var keysTransposed = keysReshaped.Transpose(1, 2);

        // Batched matmul: [batch * numHeads, numQueries, numCached]
        var scores = Tensor.Matmul(queryReshaped, keysTransposed);

        // Reshape back: [batch, numQueries, numHeads, numCached]
        return scores.Reshape(batchSize, numQueries, numHeads, numCached);
    }

    private Tensor MatmulWeightValue(
        Tensor weights,
        Tensor values,
        int batchSize,
        int numQueries,
        int numHeads,
        int numCached,
        int headDim)
    {
        // Weights: [batch, numQueries, numHeads, numCached]
        // Values:  [batch, numCached, numHeads, headDim]
        // Output: [batch, numQueries, numHeads, headDim]

        // Reshape for batched matmul
        // Weights -> [batch * numHeads, numQueries, numCached]
        var weightsReshaped = weights.Reshape(
            batchSize * numHeads,
            numQueries,
            numCached
        );

        // Values -> [batch * numHeads, numCached, headDim]
        var valuesReshaped = values.Reshape(
            batchSize * numHeads,
            numCached,
            headDim
        );

        // Batched matmul: [batch * numHeads, numQueries, headDim]
        var output = Tensor.Matmul(weightsReshaped, valuesReshaped);

        // Reshape back: [batch, numQueries, numHeads, headDim]
        return output.Reshape(batchSize, numQueries, numHeads, headDim);
    }

    private Tensor Softmax(Tensor input, int dim)
    {
        // Find max for numerical stability
        var max = input.Max(dim: dim, keepDim: true);

        // Subtract max and exponentiate
        var exp = (input - max).Exp();

        // Sum and normalize
        var sum = exp.Sum(dim: dim, keepDim: true);
        return exp / sum;
    }

    private Tensor ApplyCausalMask(
        Tensor scores,
        int batchSize,
        int numQueries,
        int numHeads,
        int numCached)
    {
        // Create mask: [1, 1, 1, numCached]
        // Mask out future tokens (positions >= numCached - numQueries)
        var mask = Tensor.Ones(new[] { 1, 1, 1, numCached }, scores.Device);

        // Set invalid positions to -inf
        int startPos = numCached - numQueries;
        for (int i = startPos + 1; i < numCached; i++)
        {
            mask[0, 0, 0, i] = float.NegativeInfinity;
        }

        // Apply mask to scores
        return scores + mask;
    }

    public bool SupportsDevice(Device device)
    {
        // Standard kernel works on CPU and GPU (though not optimized)
        return true;
    }
}

/// <summary>
/// Execution phase for attention computation.
/// </summary>
public enum AttentionPhase
{
    /// <summary>
    /// Prefill phase: processing prompt tokens in parallel.
    /// </summary>
    Prefill,

    /// <summary>
    /// Decode phase: processing generated tokens one at a time.
    /// </summary>
    Decode
}
```

### CudaPagedAttentionKernel (Placeholder for Optimized Implementation)
```csharp
namespace MlFramework.Inference.PagedAttention.Kernels;

/// <summary>
/// CUDA-optimized implementation of paged attention.
/// This is a placeholder for a future optimized CUDA kernel implementation.
/// </summary>
public class CudaPagedAttentionKernel : IPagedAttentionKernel
{
    private readonly int _headDim;
    private readonly IntPtr _kernelHandle;

    public CudaPagedAttentionKernel(int headDim)
    {
        _headDim = headDim;
        // Initialize CUDA kernel
        _kernelHandle = InitializeCudaKernel(headDim);
    }

    public Tensor ComputePagedAttention(
        Tensor query,
        Tensor cachedKeys,
        Tensor cachedValues,
        AttentionPhase phase,
        float scale = -1.0f)
    {
        // TODO: Implement CUDA kernel call
        // This would use custom CUDA kernels for:
        // 1. FlashAttention-style computation
        // 2. Block-sparse attention patterns
        // 3. Tensor cores optimization
        throw new NotImplementedException("CUDA kernel not yet implemented");
    }

    public bool SupportsDevice(Device device)
    {
        return device.Type == DeviceType.CUDA;
    }

    private IntPtr InitializeCudaKernel(int headDim)
    {
        // Load CUDA kernels from PTX/Cubin files
        // Compile kernels at runtime or load pre-compiled binaries
        return IntPtr.Zero;
    }
}
```

## Requirements
1. **Correctness**: Standard kernel must produce correct attention outputs
2. **Numerical Stability**: Handle softmax overflow/underflow properly
3. **Phase Awareness**: Different optimizations for prefill vs decode
4. **Device Support**: Proper device detection and fallback mechanisms
5. **Extensibility**: Interface allows for optimized hardware-specific implementations

## Testing Requirements
1. Unit tests for standard kernel correctness
2. Unit tests for softmax numerical stability
3. Unit tests for causal mask application
4. Comparison tests with existing attention implementation
5. Device support tests

## Estimated Time
45-60 minutes

## Dependencies
- spec_pagedattention_models.md

## Success Criteria
- Correct attention computation
- Numerically stable softmax
- Proper phase handling
- Device detection works
- Interface is clear and extensible
- Standard implementation passes correctness tests
