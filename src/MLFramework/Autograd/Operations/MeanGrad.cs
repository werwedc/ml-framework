using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd.Operations;

/// <summary>
/// Gradient computation for mean reduction operation.
/// For z = mean(x), dz/dx = 1/n for all elements (broadcasted back to input shape).
/// </summary>
public class MeanGrad : IOperationGrad
{
    /// <inheritdoc/>
    public string OperationName => "Mean";

    /// <inheritdoc/>
    public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
    {
        if (inputs.Length < 1)
            throw new ArgumentException("Mean operation requires at least 1 input", nameof(inputs));

        var x = inputs[0];

        // d(mean(x))/dx = 1/n for each element, where n is the number of elements
        int n = x.Size;
        float scaleFactor = 1.0f / n;

        // The gradient is gradOutput * (1/n) broadcasted to the input shape
        var gradX = BroadcastAndScale(gradOutput, x.Shape, scaleFactor);

        return new Tensor[] { gradX };
    }

    /// <summary>
    /// Broadcasts a tensor to the target shape and scales it.
    /// </summary>
    private static Tensor BroadcastAndScale(Tensor tensor, int[] targetShape, float scaleFactor)
    {
        if (AreShapesEqual(tensor.Shape, targetShape))
        {
            var resultData = new float[tensor.Size];
            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = tensor.Data[i] * scaleFactor;
            }
            return new Tensor(resultData, targetShape);
        }

        var targetSize = targetShape.Aggregate(1, (x, y) => x * y);
        var scaledData = new float[targetSize];

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

            scaledData[i] = tensor.Data[sourceIdx] * scaleFactor;
        }

        return new Tensor(scaledData, targetShape);
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
