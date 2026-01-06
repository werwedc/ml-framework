using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd.Operations;

/// <summary>
/// Gradient computation for exponential operation.
/// For z = exp(x), dz/dx = exp(x) = z.
/// </summary>
public class ExpGrad : IOperationGrad
{
    /// <inheritdoc/>
    public string OperationName => "Exp";

    /// <inheritdoc/>
    public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
    {
        if (inputs.Length < 1)
            throw new ArgumentException("Exp operation requires at least 1 input", nameof(inputs));

        var x = inputs[0];

        // d(exp(x))/dx = exp(x) = output
        // We need to retrieve the output from the context or compute it
        Tensor output;
        if (context.HasSavedTensor("output"))
        {
            output = context.GetSavedTensor<Tensor>("output");
        }
        else
        {
            // Fallback: compute exp(x) if not saved
            output = ElementwiseExp(x);
        }

        // Gradient = gradOutput * output
        var gradX = ElementwiseMul(gradOutput, output);

        return new Tensor[] { gradX };
    }

    /// <summary>
    /// Computes element-wise exponential.
    /// </summary>
    private static Tensor ElementwiseExp(Tensor tensor)
    {
        var newData = new float[tensor.Size];

        for (int i = 0; i < newData.Length; i++)
        {
            // Clamp to avoid overflow
            if (tensor.Data[i] > 88.0f) // exp(88) is near float max
            {
                newData[i] = float.MaxValue;
            }
            else
            {
                newData[i] = (float)Math.Exp(tensor.Data[i]);
            }
        }

        return new Tensor(newData, tensor.Shape);
    }

    /// <summary>
    /// Element-wise multiplication of two tensors.
    /// </summary>
    private static Tensor ElementwiseMul(Tensor a, Tensor b)
    {
        if (AreShapesEqual(a.Shape, b.Shape))
        {
            var resultData = new float[a.Size];
            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] * b.Data[i];
            }
            return new Tensor(resultData, a.Shape);
        }

        var broadcastedShape = BroadcastShape(a.Shape, b.Shape);
        var aBroadcasted = BroadcastTo(a, broadcastedShape);
        var bBroadcasted = BroadcastTo(b, broadcastedShape);

        var resultData2 = new float[broadcastedShape.Aggregate(1, (x, y) => x * y)];
        for (int i = 0; i < resultData2.Length; i++)
        {
            resultData2[i] = aBroadcasted.Data[i] * bBroadcasted.Data[i];
        }

        return new Tensor(resultData2, broadcastedShape);
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
