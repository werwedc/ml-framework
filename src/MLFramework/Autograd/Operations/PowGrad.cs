using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd.Operations;

/// <summary>
/// Gradient computation for power operation.
/// For z = x^n, dz/dx = n * x^(n-1).
/// </summary>
public class PowGrad : IOperationGrad
{
    /// <inheritdoc/>
    public string OperationName => "Pow";

    /// <inheritdoc/>
    public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
    {
        if (inputs.Length < 2)
            throw new ArgumentException("Pow operation requires at least 2 inputs", nameof(inputs));

        var x = inputs[0];
        var n = inputs[1]; // This can be a scalar tensor

        // d(x^n)/dx = n * x^(n-1)
        // Assuming n is a constant (doesn't require gradient)
        var gradX = ComputePowerGradient(gradOutput, x, n);

        // d(x^n)/dn = x^n * ln(x) (if n requires gradient)
        // For simplicity, we'll assume n is constant, so gradN is zero
        var gradN = Tensor.Zeros(n.Shape);

        return new Tensor[] { gradX, gradN };
    }

    /// <summary>
    /// Computes the gradient for power operation.
    /// </summary>
    private static Tensor ComputePowerGradient(Tensor gradOutput, Tensor x, Tensor n)
    {
        // n * x^(n-1)
        // First, get n as a scalar value
        float nValue = GetScalarValue(n);

        // Compute x^(n-1)
        float exponent = nValue - 1.0f;
        var xToNMinus1 = ElementwisePow(x, exponent);

        // Multiply by n
        var factor = nValue;

        // Element-wise multiply: n * x^(n-1)
        var factorTensor = Tensor.Ones(x.Shape);
        for (int i = 0; i < factorTensor.Size; i++)
        {
            factorTensor.Data[i] = factor;
        }

        var gradX = ElementwiseMul(factorTensor, xToNMinus1);

        // Multiply by gradOutput
        return ElementwiseMul(gradX, gradOutput);
    }

    /// <summary>
    /// Gets the scalar value from a tensor (assuming it's a 0-d or single-element tensor).
    /// </summary>
    private static float GetScalarValue(Tensor tensor)
    {
        if (tensor.Size != 1)
            throw new ArgumentException("Power operation expects exponent to be a scalar", nameof(tensor));

        return tensor.Data[0];
    }

    /// <summary>
    /// Computes element-wise power with a scalar exponent.
    /// </summary>
    private static Tensor ElementwisePow(Tensor tensor, float exponent)
    {
        var newData = new float[tensor.Size];

        for (int i = 0; i < newData.Length; i++)
        {
            if (tensor.Data[i] < 0 && !IsInteger(exponent))
            {
                // Handle negative base with non-integer exponent
                // Use absolute value and return NaN (similar to PyTorch)
                newData[i] = float.NaN;
            }
            else
            {
                newData[i] = (float)Math.Pow(tensor.Data[i], exponent);
            }
        }

        return new Tensor(newData, tensor.Shape);
    }

    /// <summary>
    /// Checks if a float is close to an integer value.
    /// </summary>
    private static bool IsInteger(float value)
    {
        return Math.Abs(value - Math.Round(value)) < 1e-6f;
    }

    /// <summary>
    /// Element-wise multiplication of two tensors.
    /// </summary>
    private static Tensor ElementwiseMul(Tensor a, Tensor b)
    {
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
            var multiIdx = FlatToMultiIndex(i, strides);

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

        var paddedOriginalShape = new int[nDimsGrad];
        for (int i = 0; i < nDimsGrad; i++)
        {
            int originalIdx = i - (nDimsGrad - nDimsOriginal);
            paddedOriginalShape[i] = (originalIdx >= 0) ? originalShape[originalIdx] : 1;
        }

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

        var newShape = new int[shape.Length - 1];
        int newIdx = 0;
        for (int i = 0; i < shape.Length; i++)
        {
            if (i != axis)
            {
                newShape[newIdx++] = shape[i];
            }
        }

        var newData = new float[newShape.Aggregate(1, (a, b) => a * b)];

        var strides = ComputeStrides(shape);
        var newStrides = ComputeStrides(newShape);

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
