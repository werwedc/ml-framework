using System;
using System.Collections.Generic;
using System.Linq;
using RitterFramework.Core.Tensor;

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
        var rank = tensors[0].Dimensions;
        foreach (var tensor in tensors)
        {
            if (tensor.Dimensions != rank)
                throw new ArgumentException("All tensors must have the same rank", nameof(tensors));
        }

        // Find maximum dimensions
        var maxDims = GetMaxDimensions(tensors);

        // Create stacked tensor
        var batchSize = tensors.Count;
        var stackedShape = new int[rank + 1];
        stackedShape[0] = batchSize;
        Array.Copy(maxDims, 0, stackedShape, 1, rank);

        var stackedTensor = Tensor.Zeros(stackedShape);

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

    private static int[] GetMaxDimensions(List<Tensor> tensors)
    {
        var rank = tensors[0].Dimensions;
        var maxDims = new int[rank];

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
        int[] maxDims,
        float paddingValue,
        PaddingStrategy strategy)
    {
        var rank = sourceTensor.Dimensions;
        var batchSize = stackedTensor.Shape[0];

        // Get the maximum sequence length (dimension 0 of each tensor)
        var maxSeqLen = maxDims[0];

        // Copy source tensor data to batch position
        if (strategy == PaddingStrategy.Post)
        {
            // Post-padding: copy first, then pad
            for (int i = 0; i < sourceTensor.Shape[0]; i++)
            {
                for (int j = 0; j < sourceTensor.Size / sourceTensor.Shape[0]; j++)
                {
                    int[] stackedIndices = new int[rank + 1];
                    stackedIndices[0] = batchIndex;
                    stackedIndices[1] = i;

                    // Handle multi-dimensional tensors
                    if (rank > 1)
                    {
                        var sourceIndices = GetIndicesForFlatIndex(sourceTensor, i * (sourceTensor.Size / sourceTensor.Shape[0]) + j);
                        for (int k = 0; k < rank - 1; k++)
                        {
                            stackedIndices[2 + k] = sourceIndices[k + 1];
                        }
                    }

                    stackedTensor[stackedIndices] = sourceTensor[GetIndicesForFlatIndex(sourceTensor, i * (sourceTensor.Size / sourceTensor.Shape[0]) + j)];
                }
            }

            // Fill remaining with padding
            for (int i = sourceTensor.Shape[0]; i < maxSeqLen; i++)
            {
                for (int j = 0; j < GetTotalElementsForDims(maxDims, 1); j++)
                {
                    int[] stackedIndices = new int[rank + 1];
                    stackedIndices[0] = batchIndex;
                    stackedIndices[1] = i;

                    if (rank > 1)
                    {
                        var paddingIndices = GetIndicesForFlatIndex(new Tensor(new float[GetTotalElementsForDims(maxDims, 1)], new int[rank - 1]), j);
                        for (int k = 0; k < rank - 1; k++)
                        {
                            stackedIndices[2 + k] = paddingIndices[k];
                        }
                    }

                    stackedTensor[stackedIndices] = paddingValue;
                }
            }
        }
        else
        {
            // Pre-padding: pad first, then copy
            // Fill padding positions
            for (int i = 0; i < maxSeqLen - sourceTensor.Shape[0]; i++)
            {
                for (int j = 0; j < GetTotalElementsForDims(maxDims, 1); j++)
                {
                    int[] stackedIndices = new int[rank + 1];
                    stackedIndices[0] = batchIndex;
                    stackedIndices[1] = i;

                    if (rank > 1)
                    {
                        var paddingIndices = GetIndicesForFlatIndex(new Tensor(new float[GetTotalElementsForDims(maxDims, 1)], new int[rank - 1]), j);
                        for (int k = 0; k < rank - 1; k++)
                        {
                            stackedIndices[2 + k] = paddingIndices[k];
                        }
                    }

                    stackedTensor[stackedIndices] = paddingValue;
                }
            }

            // Copy source tensor data
            for (int i = 0; i < sourceTensor.Shape[0]; i++)
            {
                for (int j = 0; j < sourceTensor.Size / sourceTensor.Shape[0]; j++)
                {
                    int[] stackedIndices = new int[rank + 1];
                    stackedIndices[0] = batchIndex;
                    stackedIndices[1] = i + maxSeqLen - sourceTensor.Shape[0];

                    if (rank > 1)
                    {
                        var sourceIndices = GetIndicesForFlatIndex(sourceTensor, i * (sourceTensor.Size / sourceTensor.Shape[0]) + j);
                        for (int k = 0; k < rank - 1; k++)
                        {
                            stackedIndices[2 + k] = sourceIndices[k + 1];
                        }
                    }

                    stackedTensor[stackedIndices] = sourceTensor[GetIndicesForFlatIndex(sourceTensor, i * (sourceTensor.Size / sourceTensor.Shape[0]) + j)];
                }
            }
        }
    }

    private static int[] GetIndicesForFlatIndex(Tensor tensor, int flatIndex)
    {
        var indices = new int[tensor.Dimensions];
        var remaining = flatIndex;

        for (int i = 0; i < tensor.Dimensions; i++)
        {
            var stride = 1;
            for (int j = i + 1; j < tensor.Dimensions; j++)
            {
                stride *= tensor.Shape[j];
            }

            indices[i] = remaining / stride;
            remaining = remaining % stride;
        }

        return indices;
    }

    private static int GetTotalElementsForDims(int[] shape, int startDim)
    {
        int total = 1;
        for (int i = startDim; i < shape.Length; i++)
        {
            total *= shape[i];
        }
        return total;
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
        if (stackedTensor.Dimensions < 2)
            throw new ArgumentException("Stacked tensor must have at least 2 dimensions");

        // Determine the shape of the output tensor
        var outputShape = new int[stackedTensor.Dimensions - 1];
        outputShape[0] = length;
        for (int i = 1; i < outputShape.Length; i++)
        {
            outputShape[i] = stackedTensor.Shape[i + 1];
        }

        var outputTensor = Tensor.Zeros(outputShape);

        // Copy data from stacked tensor to output tensor
        for (int i = 0; i < length; i++)
        {
            for (int j = 0; j < GetTotalElementsForDims(outputShape, 1); j++)
            {
                // Build indices for stacked tensor
                var stackedIndices = new int[stackedTensor.Dimensions];
                stackedIndices[0] = batchIndex;
                stackedIndices[1] = i;

                if (stackedTensor.Dimensions > 2)
                {
                    var otherIndices = GetIndicesForFlatIndex(outputTensor, i * GetTotalElementsForDims(outputShape, 1) + j);
                    for (int k = 1; k < otherIndices.Length; k++)
                    {
                        stackedIndices[k + 1] = otherIndices[k];
                    }
                }

                // Build indices for output tensor
                var outputIndices = GetIndicesForFlatIndex(outputTensor, i * GetTotalElementsForDims(outputShape, 1) + j);

                outputTensor[outputIndices] = stackedTensor[stackedIndices];
            }
        }

        return outputTensor;
    }
}
