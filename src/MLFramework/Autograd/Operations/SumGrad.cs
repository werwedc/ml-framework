using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd.Operations;

/// <summary>
/// Gradient computation for sum reduction operation.
/// For z = sum(x), dz/dx = 1 for all elements (broadcasted back to input shape).
/// </summary>
public class SumGrad : IOperationGrad
{
    /// <inheritdoc/>
    public string OperationName => "Sum";

    /// <inheritdoc/>
    public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
    {
        if (inputs.Length < 1)
            throw new ArgumentException("Sum operation requires at least 1 input", nameof(inputs));

        var x = inputs[0];

        // d(sum(x))/dx = 1 for each element
        // The gradient is just gradOutput broadcasted to the input shape
        var gradX = BroadcastTo(gradOutput, x.Shape);

        return new Tensor[] { gradX };
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
