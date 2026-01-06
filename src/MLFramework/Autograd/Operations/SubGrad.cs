using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd.Operations;

/// <summary>
/// Gradient computation for subtraction operation.
/// For z = x - y, dz/dx = 1 and dz/dy = -1.
/// </summary>
public class SubGrad : IOperationGrad
{
    /// <inheritdoc/>
    public string OperationName => "Sub";

    /// <inheritdoc/>
    public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
    {
        // d(x - y)/dx = 1, d(x - y)/dy = -1
        // The gradient for first input passes through unchanged
        // The gradient for second input passes through with sign flip
        var gradX = gradOutput.Clone();
        var gradY = Negate(gradOutput);

        // Handle broadcasting if necessary
        if (inputs.Length >= 2 && !AreShapesEqual(inputs[0].Shape, inputs[1].Shape))
        {
            if (!AreShapesEqual(gradX.Shape, inputs[0].Shape))
            {
                gradX = HandleBroadcastingGrad(gradX, inputs[0].Shape);
            }

            if (!AreShapesEqual(gradY.Shape, inputs[1].Shape))
            {
                gradY = HandleBroadcastingGrad(gradY, inputs[1].Shape);
            }
        }

        return new Tensor[] { gradX, gradY };
    }

    /// <summary>
    /// Negates all elements in a tensor.
    /// </summary>
    private static Tensor Negate(Tensor tensor)
    {
        var newData = new float[tensor.Size];
        for (int i = 0; i < newData.Length; i++)
        {
            newData[i] = -tensor.Data[i];
        }
        return new Tensor(newData, tensor.Shape);
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
