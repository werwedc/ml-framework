# Spec: Tensor Operations for Batching

## Overview
Implement utilities for padding and stacking tensors to handle variable-length inputs in batches.

## Technical Requirements

### Padding Strategy Enum
```csharp
namespace MLFramework.Serving;

/// <summary>
/// Defines padding strategy for variable-length sequences
/// </summary>
public enum PaddingStrategy
{
    /// <summary>
    /// Pad at the end of sequence (post-padding)
    /// </summary>
    Post,

    /// <summary>
    /// Pad at the beginning of sequence (pre-padding)
    /// </summary>
    Pre
}
```

### Tensor Stack Result
```csharp
namespace MLFramework.Serving;

/// <summary>
/// Result of stacking tensors with padding metadata
/// </summary>
public class TensorStackResult
{
    /// <summary>
    /// The stacked and padded tensor ready for model inference
    /// </summary>
    public Tensor StackedTensor { get; }

    /// <summary>
    /// Original lengths of each tensor before padding (for debatching)
    /// </summary>
    public int[] OriginalLengths { get; }

    /// <summary>
    /// Mapping from batch index to original request index
    /// </summary>
    public int[] BatchIndices { get; }

    public TensorStackResult(Tensor stackedTensor, int[] originalLengths, int[] batchIndices)
    {
        StackedTensor = stackedTensor ?? throw new ArgumentNullException(nameof(stackedTensor));
        OriginalLengths = originalLengths ?? throw new ArgumentNullException(nameof(originalLengths));
        BatchIndices = batchIndices ?? throw new ArgumentNullException(nameof(batchIndices));
    }
}
```

### Tensor Operations Utility
```csharp
namespace MLFramework.Serving;

/// <summary>
/// Utilities for padding and stacking tensors for batching
/// </summary>
public static class TensorOperations
{
    /// <summary>
    /// Stack tensors with automatic padding to match the longest tensor
    /// </summary>
    /// <param name="tensors">List of tensors to stack</param>
    /// <param name="paddingValue">Value to use for padding (default: 0)</param>
    /// <param name="strategy">Padding strategy (default: Post)</param>
    /// <returns>Stacked tensor with metadata</returns>
    public static TensorStackResult StackWithPadding(
        List<Tensor> tensors,
        float paddingValue = 0f,
        PaddingStrategy strategy = PaddingStrategy.Post)
    {
        if (tensors == null || tensors.Count == 0)
            throw new ArgumentException("Tensors list cannot be null or empty", nameof(tensors));

        // Validate all tensors have same rank
        var rank = tensors[0].Rank;
        foreach (var tensor in tensors)
        {
            if (tensor.Rank != rank)
                throw new ArgumentException("All tensors must have the same rank", nameof(tensors));
        }

        // Find maximum dimensions
        var maxDims = GetMaxDimensions(tensors);

        // Create stacked tensor
        var batchSize = tensors.Count;
        var stackedShape = new long[rank + 1];
        stackedShape[0] = batchSize;
        Array.Copy(maxDims, 0, stackedShape, 1, rank);

        var stackedTensor = new Tensor(stackedShape);

        // Track original lengths
        var originalLengths = new int[tensors.Count];

        // Fill stacked tensor
        for (int i = 0; i < tensors.Count; i++)
        {
            var tensor = tensors[i];
            originalLengths[i] = tensor.Shape[0]; // Assume sequence length is dim 0

            CopyWithPadding(stackedTensor, i, tensor, maxDims, paddingValue, strategy);
        }

        var batchIndices = Enumerable.Range(0, tensors.Count).ToArray();

        return new TensorStackResult(stackedTensor, originalLengths, batchIndices);
    }

    private static long[] GetMaxDimensions(List<Tensor> tensors)
    {
        var rank = tensors[0].Rank;
        var maxDims = new long[rank];

        for (int d = 0; d < rank; d++)
        {
            maxDims[d] = tensors.Max(t => t.Shape[d]);
        }

        return maxDims;
    }

    private static void CopyWithPadding(
        Tensor stackedTensor,
        int batchIndex,
        Tensor sourceTensor,
        long[] maxDims,
        float paddingValue,
        PaddingStrategy strategy)
    {
        // Implementation depends on tensor library API
        // This is a placeholder showing the logic
        // Copy source tensor values to batch position
        // Fill remaining positions with paddingValue
    }

    /// <summary>
    /// Slice stacked tensor back to individual tensors based on original lengths
    /// </summary>
    /// <param name="stackedTensor">The stacked tensor from model output</param>
    /// <param name="originalLengths">Original lengths of each item</param>
    /// <returns>List of individual tensors</returns>
    public static List<Tensor> Unstack(Tensor stackedTensor, int[] originalLengths)
    {
        if (stackedTensor == null)
            throw new ArgumentNullException(nameof(stackedTensor));

        if (originalLengths == null || originalLengths.Length == 0)
            throw new ArgumentException("Original lengths cannot be null or empty", nameof(originalLengths));

        if (originalLengths.Length != stackedTensor.Shape[0])
            throw new ArgumentException("Original lengths count must match batch size", nameof(originalLengths));

        var result = new List<Tensor>();

        for (int i = 0; i < originalLengths.Length; i++)
        {
            var sliced = SliceBatchItem(stackedTensor, i, originalLengths[i]);
            result.Add(sliced);
        }

        return result;
    }

    private static Tensor SliceBatchItem(Tensor stackedTensor, int batchIndex, int length)
    {
        // Implementation depends on tensor library API
        // Slice the tensor at batch index up to original length
        throw new NotImplementedException();
    }
}
```

## File Location
- **Path:** `src/Serving/TensorOperations.cs`

## Dependencies
- Framework tensor library (Tensor, basic operations)

## Key Design Decisions

1. **Simple MVP**: Focus on 1D sequence tensors initially
2. **Post-Padding Default**: Most common use case for NLP models
3. **Metadata Tracking**: Original lengths preserved for accurate debatching
4. **Validation**: Ensure consistent tensor ranks before stacking

## Success Criteria
- Stacking handles variable-length tensors correctly
- Padding is applied according to strategy
- Original lengths are accurately tracked
- Unstacking reverses stacking operation correctly
- Appropriate validation for edge cases

## Testing Requirements
- Test stacking same-length tensors (no padding)
- Test stacking variable-length tensors with post-padding
- Test stacking variable-length tensors with pre-padding
- Test custom padding value
- Test unstacking restores original tensors
- Test validation with mismatched tensor ranks
- Test validation with empty tensor list
- Test batch size dimension in stacked tensor

## Notes
This is an MVP implementation. Future enhancements could include:
- Multi-dimensional padding strategies
- Attention mask generation
- Zero-copy optimizations
