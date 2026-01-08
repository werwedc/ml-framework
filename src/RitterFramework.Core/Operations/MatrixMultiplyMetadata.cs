using System;
using System.Linq;

namespace RitterFramework.Core.Operations;

/// <summary>
/// Metadata for matrix multiplication operations.
/// Validates that inner dimensions match: [M, K] × [K, N]
/// </summary>
public class MatrixMultiplyMetadata : IOperationMetadata
{
    /// <inheritdoc/>
    public OperationType Type => OperationType.MatrixMultiply;

    /// <inheritdoc/>
    public string Name => "MatrixMultiply";

    /// <inheritdoc/>
    public int RequiredInputTensors => 2;

    /// <inheritdoc/>
    public bool ValidateInputShapes(params long[][] inputShapes)
    {
        if (inputShapes.Length != 2)
        {
            return false;
        }

        var shape1 = inputShapes[0];
        var shape2 = inputShapes[1];

        // Both tensors should be at least 2D for matrix multiplication
        if (shape1.Length < 2 || shape2.Length < 2)
        {
            return false;
        }

        // Get the last two dimensions
        // For shape1: [..., M, K]
        // For shape2: [..., K, N]
        long dim1_K = shape1[shape1.Length - 1];
        long dim2_K = shape2[shape2.Length - 2];

        // Inner dimensions must match
        return dim1_K == dim2_K;
    }

    /// <inheritdoc/>
    public long[] InferOutputShape(params long[][] inputShapes)
    {
        var shape1 = inputShapes[0];
        var shape2 = inputShapes[1];

        // Get the last two dimensions
        // For shape1: [..., M, K] -> output: [..., M, N]
        // For shape2: [..., K, N]
        long M = shape1[shape1.Length - 2];
        long N = shape2[shape2.Length - 1];

        // Broadcast leading dimensions if they exist
        int maxRank = Math.Max(shape1.Length, shape2.Length);

        if (maxRank <= 2)
        {
            return new long[] { M, N };
        }

        // For higher rank tensors, we need to broadcast leading dimensions
        var outputShape = new long[maxRank];
        outputShape[maxRank - 2] = M;
        outputShape[maxRank - 1] = N;

        // Copy leading dimensions from the larger tensor
        for (int i = 0; i < maxRank - 2; i++)
        {
            if (i < shape1.Length - 2 && i < shape2.Length - 2)
            {
                // Both tensors have this dimension - they must be compatible for broadcasting
                long dim1 = shape1[shape1.Length - 3 - i];
                long dim2 = shape2[shape2.Length - 3 - i];

                if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
                {
                    throw new ArgumentException(
                        $"Cannot broadcast shapes [{string.Join(", ", shape1)}] and [{string.Join(", ", shape2)}]");
                }

                outputShape[maxRank - 3 - i] = Math.Max(dim1, dim2);
            }
            else if (i < shape1.Length - 2)
            {
                outputShape[maxRank - 3 - i] = shape1[shape1.Length - 3 - i];
            }
            else
            {
                outputShape[maxRank - 3 - i] = shape2[shape2.Length - 3 - i];
            }
        }

        return outputShape;
    }
}
