using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd.Operations;

/// <summary>
/// Gradient computation for multiplication operation.
/// For z = x * y, dz/dx = y and dz/dy = x.
/// </summary>
public class MulGrad : IOperationGrad
{
    /// <inheritdoc/>
    public string OperationName => "Mul";

    /// <inheritdoc/>
    public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
    {
        if (inputs.Length < 2)
            throw new ArgumentException("Mul operation requires at least 2 inputs", nameof(inputs));

        var x = inputs[0];
        var y = inputs[1];

        // d(x * y)/dx = y, d(x * y)/dy = x
        var gradX = ElementwiseMul(gradOutput, y);
        var gradY = ElementwiseMul(gradOutput, x);

        // Handle broadcasting if necessary
        if (!AreShapesEqual(gradX.Shape, x.Shape))
        {
            gradX = HandleBroadcastingGrad(gradX, x.Shape);
        }

        if (!AreShapesEqual(gradY.Shape, y.Shape))
        {
            gradY = HandleBroadcastingGrad(gradY, y.Shape);
        }

        return new Tensor[] { gradX, gradY };
    }

    /// <summary>
    /// Checks if two shapes are equal.
    /// </summary>
    private static bool AreShapesEqual(int[] shape1, int[] shape2)
    {
        if (shape1.Length != shape2.Length)
            return false;

        for (int i = 0; i < shape1.Length; i++)
        {
            if (shape1[i] != shape2[i])
                return false;
        }

        return true;
    }

    /// <summary>
    /// Element-wise multiplication of two tensors.
    /// </summary>
    private static Tensor ElementwiseMul(Tensor a, Tensor b)
    {
        // Ensure tensors are broadcast-compatible
        var broadcastedShape = BroadcastShape(a.Shape, b.Shape);
        var aBroadcasted = BroadcastTo(a, broadcastedShape);
        var bBroadcasted = BroadcastTo(b, broadcastedShape);

        var resultData = new float[broadcastedShape.Aggregate(1, (x, y) => x * y)];
        for (int i = 0; i < resultData.Length; i++)
        {
            resultData[i] = aBroadcasted.Data[i] * bBroadcasted.Data[i];
        }

        return new Tensor(resultData, broadcastedShape);
    }

    /// <summary>
    /// Broadcasts a tensor to the target shape.
    /// </summary>
    private static Tensor BroadcastTo(Tensor tensor, int[] targetShape)
    {
        if (AreShapesEqual(tensor.Shape, targetShape))
            return tensor.Clone();

        var targetSize = targetShape.Aggregate(1, (x, y) => x * y);
        var resultData = new float[targetSize];

        var strides = ComputeStrides(targetShape);
        var tensorStrides = ComputeStrides(tensor.Shape);

        // Pad tensor shape with leading 1s
        var paddedTensorShape = new int[targetShape.Length];
        int padAmount = targetShape.Length - tensor.Shape.Length;
        for (int i = 0; i < targetShape.Length; i++)
        {
            paddedTensorShape[i] = (i >= padAmount) ? tensor.Shape[i - padAmount] : 1;
        }

        var paddedTensorStrides = ComputeStrides(paddedTensorShape);

        for (int i = 0; i < targetSize; i++)
        {
            // Get multi-index in target
            var multiIdx = FlatToMultiIndex(i, strides);

            // Map to source index (handling broadcasting)
            int sourceIdx = 0;
            for (int j = 0; j < targetShape.Length; j++)
            {
                int idxInSource = (paddedTensorShape[j] == 1) ? 0 : multiIdx[j];
                sourceIdx += idxInSource * paddedTensorStrides[j];
            }

            resultData[i] = tensor.Data[sourceIdx];
        }

        return new Tensor(resultData, targetShape);
    }

    /// <summary>
    /// Computes the broadcast shape for two tensors.
    /// </summary>
    private static int[] BroadcastShape(int[] shape1, int[] shape2)
    {
        int maxDims = Math.Max(shape1.Length, shape2.Length);
        var result = new int[maxDims];

        for (int i = 0; i < maxDims; i++)
        {
            int dim1 = (i < maxDims - shape1.Length) ? 1 : shape1[i - (maxDims - shape1.Length)];
            int dim2 = (i < maxDims - shape2.Length) ? 1 : shape2[i - (maxDims - shape2.Length)];

            if (dim1 == dim2 || dim1 == 1 || dim2 == 1)
            {
                result[i] = Math.Max(dim1, dim2);
            }
            else
            {
                throw new ArgumentException($"Shapes {string.Join(",", shape1)} and {string.Join(",", shape2)} are not broadcast-compatible");
            }
        }

        return result;
    }

    /// <summary>
    /// Handles gradient reduction when broadcasting was used.
    /// </summary>
    private static Tensor HandleBroadcastingGrad(Tensor grad, int[] originalShape)
    {
        var gradShape = grad.Shape;
        var nDimsGrad = gradShape.Length;
        var nDimsOriginal = originalShape.Length;

        // Pad original shape with leading 1s if necessary
        var paddedOriginalShape = new int[nDimsGrad];
        for (int i = 0; i < nDimsGrad; i++)
        {
            int originalIdx = i - (nDimsGrad - nDimsOriginal);
            paddedOriginalShape[i] = (originalIdx >= 0) ? originalShape[originalIdx] : 1;
        }

        // Sum over broadcasted dimensions
        var result = grad.Clone();
        for (int i = 0; i < nDimsGrad; i++)
        {
            if (paddedOriginalShape[i] == 1 && gradShape[i] > 1)
            {
                result = SumAlongAxis(result, i);
            }
        }

        return result;
    }

    /// <summary>
    /// Sums a tensor along a specific axis.
    /// </summary>
    private static Tensor SumAlongAxis(Tensor tensor, int axis)
    {
        var shape = tensor.Shape;
        if (axis < 0 || axis >= shape.Length)
            throw new ArgumentOutOfRangeException(nameof(axis));

        // Compute new shape (remove the axis being summed)
        var newShape = new int[shape.Length - 1];
        int newIdx = 0;
        for (int i = 0; i < shape.Length; i++)
        {
            if (i != axis)
            {
                newShape[newIdx++] = shape[i];
            }
        }

        // Compute sum along the axis
        var newData = new float[newShape.Aggregate(1, (a, b) => a * b)];

        // Compute strides
        var strides = ComputeStrides(shape);
        var newStrides = ComputeStrides(newShape);

        // Sum along the axis
        for (int i = 0; i < newData.Length; i++)
        {
            var multiIdx = FlatToMultiIndex(i, newStrides);

            float sum = 0;
            for (int k = 0; k < shape[axis]; k++)
            {
                var originalIdx = new int[shape.Length];
                int origIdx = 0;
                for (int j = 0; j < shape.Length; j++)
                {
                    if (j < axis)
                    {
                        originalIdx[j] = multiIdx[j];
                    }
                    else if (j == axis)
                    {
                        originalIdx[j] = k;
                    }
                    else
                    {
                        originalIdx[j] = multiIdx[j - 1];
                    }
                }

                int flatIdx = 0;
                for (int j = 0; j < shape.Length; j++)
                {
                    flatIdx += originalIdx[j] * strides[j];
                }

                sum += tensor.Data[flatIdx];
            }

            newData[i] = sum;
        }

        return new Tensor(newData, newShape);
    }

    private static int[] ComputeStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        strides[shape.Length - 1] = 1;
        for (int i = shape.Length - 2; i >= 0; i--)
        {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    private static int[] FlatToMultiIndex(int flatIdx, int[] strides)
    {
        var multiIdx = new int[strides.Length];
        for (int i = 0; i < strides.Length; i++)
        {
            multiIdx[i] = flatIdx / strides[i];
            flatIdx %= strides[i];
        }
        return multiIdx;
    }
}
