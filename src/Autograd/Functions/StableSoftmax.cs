using MLFramework.Autograd;
using RitterFramework.Core.Tensor;
using System;

namespace MLFramework.Autograd.Functions;

/// <summary>
/// Computes softmax with numerical stability by subtracting the maximum value before exponentiation.
/// This prevents overflow/underflow in the exponential computation.
/// </summary>
public class StableSoftmax : CustomFunction
{
    private readonly int _dim;
    private readonly bool _keepDim;

    /// <summary>
    /// Creates a new StableSoftmax instance.
    /// </summary>
    /// <param name="dim">Dimension along which to compute softmax (default: -1 for last dimension).</param>
    /// <param name="keepDim">Whether to keep reduced dimensions (default: true).</param>
    public StableSoftmax(int dim = -1, bool keepDim = true)
    {
        _dim = dim;
        _keepDim = keepDim;
    }

    /// <summary>
    /// Computes the forward pass of the stable softmax.
    /// </summary>
    /// <param name="inputs">Input tensors: [logits].</param>
    /// <param name="ctx">Function context for saving state for backward pass.</param>
    /// <returns>Array containing the softmax probabilities.</returns>
    /// <exception cref="ArgumentNullException">Thrown when input tensor is null.</exception>
    /// <exception cref="ArgumentException">Thrown when inputs array is null or empty.</exception>
    public override Tensor[] Forward(Tensor[] inputs, FunctionContext ctx)
    {
        if (inputs == null || inputs.Length != 1)
        {
            throw new ArgumentException("StableSoftmax requires exactly 1 input tensor [logits]");
        }

        var x = inputs[0];
        if (x == null)
        {
            throw new ArgumentNullException(nameof(x), "Input tensor cannot be null");
        }

        // Compute max along dimension for numerical stability
        var max = MaxAlongDim(x, _dim, true);
        
        // Subtract max: x - max(x) (with broadcasting)
        var xMinusMax = SubtractBroadcast(x, max);
        
        // Apply exponential: exp(x - max(x))
        var exp = Exp(xMinusMax);
        
        // Sum along dimension
        var sum = SumAlongDim(exp, _dim, true);
        
        // Divide: exp / sum (with broadcasting)
        var y = DivideBroadcast(exp, sum);

        // Handle keepDim=false: squeeze out the reduced dimension
        if (!_keepDim)
        {
            y = SqueezeDim(y, _dim);
        }

        // Save output for backward pass (needed for gradient computation)
        ctx.SaveForBackward(y);

        return new[] { y };
    }

    /// <summary>
    /// Computes the backward pass of the stable softmax.
    /// </summary>
    /// <param name="gradOutputs">Gradients with respect to outputs [grad_y].</param>
    /// <param name="ctx">Function context containing saved state from forward pass.</param>
    /// <returns>Array containing gradients [grad_x].</returns>
    /// <exception cref="ArgumentNullException">Thrown when gradient tensor is null.</exception>
    /// <exception cref="ArgumentException">Thrown when gradient inputs array is null or empty.</exception>
    public override Tensor[] Backward(Tensor[] gradOutputs, FunctionContext ctx)
    {
        if (gradOutputs == null)
        {
            throw new ArgumentNullException(nameof(gradOutputs), "Gradient outputs array cannot be null");
        }

        if (gradOutputs.Length != 1)
        {
            throw new ArgumentException("StableSoftmax Backward requires exactly 1 gradient output");
        }

        var grad_y = gradOutputs[0];
        if (grad_y == null)
        {
            throw new ArgumentNullException(nameof(grad_y), "Gradient tensor cannot be null");
        }

        var y = ctx.GetSavedTensor(0);

        // Gradient: grad_x = y * (grad_y - sum(grad_y * y, dim=dim, keepdim=True))
        // This is the standard softmax derivative: dy_i/dx_j = y_i * (δ_ij - y_j)
        var grad_y_y = MultiplyBroadcast(grad_y, y);
        var sum = SumAlongDim(grad_y_y, _dim, true);
        var grad_x = MultiplyBroadcast(y, SubtractBroadcast(grad_y, sum));

        // Handle keepDim=false for gradient
        if (!_keepDim)
        {
            grad_x = SqueezeDim(grad_x, _dim);
        }

        return new[] { grad_x };
    }

    /// <summary>
    /// Subtracts two tensors with broadcasting support.
    /// </summary>
    private static Tensor SubtractBroadcast(Tensor a, Tensor b)
    {
        if (a.HasSameShape(b))
        {
            return a.Subtract(b);
        }

        return BroadcastBinaryOp(a, b, (x, y) => x - y);
    }

    /// <summary>
    /// Divides two tensors with broadcasting support.
    /// </summary>
    private static Tensor DivideBroadcast(Tensor a, Tensor b)
    {
        if (a.HasSameShape(b))
        {
            return a.Divide(b);
        }

        return BroadcastBinaryOp(a, b, (x, y) => x / y);
    }

    /// <summary>
    /// Multiplies two tensors with broadcasting support.
    /// </summary>
    private static Tensor MultiplyBroadcast(Tensor a, Tensor b)
    {
        if (a.HasSameShape(b))
        {
            return a.Multiply(b);
        }

        return BroadcastBinaryOp(a, b, (x, y) => x * y);
    }

    /// <summary>
    /// Applies a binary operation with broadcasting.
    /// </summary>
    private static Tensor BroadcastBinaryOp(Tensor a, Tensor b, Func<float, float, float> op)
    {
        // Broadcast both tensors to the same shape
        var broadcastShape = GetBroadcastShape(a.Shape, b.Shape);
        var broadcastA = BroadcastTo(a, broadcastShape);
        var broadcastB = BroadcastTo(b, broadcastShape);

        // Apply operation element-wise
        var resultData = new float[broadcastA.Size];
        for (int i = 0; i < broadcastA.Size; i++)
        {
            resultData[i] = op(broadcastA.Data[i], broadcastB.Data[i]);
        }

        return new Tensor(resultData, broadcastShape, a.RequiresGrad || b.RequiresGrad, a.Dtype);
    }

    /// <summary>
    /// Gets the broadcasted shape for two tensors.
    /// </summary>
    private static int[] GetBroadcastShape(int[] shapeA, int[] shapeB)
    {
        int maxRank = Math.Max(shapeA.Length, shapeB.Length);
        var resultShape = new int[maxRank];

        for (int i = 0; i < maxRank; i++)
        {
            int dimA = i < shapeA.Length ? shapeA[shapeA.Length - 1 - i] : 1;
            int dimB = i < shapeB.Length ? shapeB[shapeB.Length - 1 - i] : 1;
            int resultDim = Math.Max(dimA, dimB);

            if (dimA != 1 && dimB != 1 && dimA != resultDim)
            {
                throw new ArgumentException($"Shapes {string.Join(",", shapeA)} and {string.Join(",", shapeB)} are not broadcastable");
            }

            resultShape[maxRank - 1 - i] = resultDim;
        }

        return resultShape;
    }

    /// <summary>
    /// Broadcasts a tensor to a target shape.
    /// </summary>
    private static Tensor BroadcastTo(Tensor tensor, int[] targetShape)
    {
        if (tensor.HasSameShape(new Tensor(new float[0], targetShape)))
        {
            return tensor;
        }

        var resultData = new float[ComputeTotalSize(targetShape)];
        BroadcastData(tensor.Data, tensor.Shape, resultData, targetShape);

        return new Tensor(resultData, targetShape, tensor.RequiresGrad, tensor.Dtype);
    }

    /// <summary>
    /// Broadcasts data from source to target shape.
    /// </summary>
    private static void BroadcastData(float[] srcData, int[] srcShape, float[] dstData, int[] dstShape)
    {
        int srcRank = srcShape.Length;
        int dstRank = dstShape.Length;

        // Compute strides for both shapes
        var srcStrides = ComputeStrides(srcShape);
        var dstStrides = ComputeStrides(dstShape);

        for (int dstIdx = 0; dstIdx < dstData.Length; dstIdx++)
        {
            // Convert dstIdx to multi-dimensional indices
            var indices = UnflattenIndex(dstIdx, dstStrides);

            // Compute srcIdx by broadcasting indices
            int srcIdx = 0;
            for (int i = 0; i < srcRank; i++)
            {
                int dimIdx = dstRank - srcRank + i;
                int idx = (srcShape[i] == 1) ? 0 : indices[dimIdx];
                srcIdx += idx * srcStrides[i];
            }

            dstData[dstIdx] = srcData[srcIdx];
        }
    }

    /// <summary>
    /// Computes total size of a shape.
    /// </summary>
    private static int ComputeTotalSize(int[] shape)
    {
        int size = 1;
        foreach (int dim in shape)
        {
            size *= dim;
        }
        return size;
    }

    /// <summary>
    /// Unflattens a flat index to multi-dimensional indices.
    /// </summary>
    private static int[] UnflattenIndex(int flatIdx, int[] strides)
    {
        var indices = new int[strides.Length];
        for (int i = 0; i < strides.Length; i++)
        {
            indices[i] = (flatIdx / strides[i]) % ((i < strides.Length - 1) ? strides[i + 1] / strides[i] : strides.Length);
        }
        return indices;
    }

    /// <summary>
    /// Computes element-wise exponential.
    /// </summary>
    private static Tensor Exp(Tensor tensor)
    {
        var resultData = new float[tensor.Size];
        for (int i = 0; i < tensor.Size; i++)
        {
            resultData[i] = MathF.Exp(tensor.Data[i]);
        }
        return new Tensor(resultData, tensor.Shape, tensor.RequiresGrad, tensor.Dtype);
    }

    /// <summary>
    /// Computes maximum along a dimension.
    /// </summary>
    private static Tensor MaxAlongDim(Tensor tensor, int dim, bool keepDim)
    {
        if (tensor.Dimensions == 0)
        {
            throw new InvalidOperationException("Cannot compute max of scalar tensor");
        }

        // Handle default dimension (last dimension)
        if (dim < 0)
        {
            dim = tensor.Dimensions - 1;
        }

        if (dim >= tensor.Dimensions)
        {
            throw new ArgumentOutOfRangeException(nameof(dim));
        }

        int dimSize = tensor.Shape[dim];
        int resultSize = tensor.Size / dimSize;

        var resultData = new float[resultSize];
        var resultShape = new int[tensor.Dimensions - 1];

        // Compute result shape
        int shapeIdx = 0;
        for (int i = 0; i < tensor.Dimensions; i++)
        {
            if (i != dim)
            {
                resultShape[shapeIdx++] = tensor.Shape[i];
            }
        }

        // Compute max along dimension
        for (int i = 0; i < resultSize; i++)
        {
            float maxVal = float.NegativeInfinity;

            for (int j = 0; j < dimSize; j++)
            {
                int srcIdx = ComputeIndexForReduce(tensor.Shape, dim, i, j);
                maxVal = Math.Max(maxVal, tensor.Data[srcIdx]);
            }

            resultData[i] = maxVal;
        }

        // Always return with keepDim for easier broadcasting
        var keepDimShape = new int[tensor.Dimensions];
        for (int i = 0; i < tensor.Dimensions; i++)
        {
            keepDimShape[i] = (i == dim) ? 1 : tensor.Shape[i];
        }
        return new Tensor(resultData, keepDimShape, tensor.RequiresGrad, tensor.Dtype);
    }

    /// <summary>
    /// Computes sum along a dimension.
    /// </summary>
    private static Tensor SumAlongDim(Tensor tensor, int dim, bool keepDim)
    {
        if (tensor.Dimensions == 0)
        {
            throw new InvalidOperationException("Cannot compute sum of scalar tensor");
        }

        // Handle default dimension (last dimension)
        if (dim < 0)
        {
            dim = tensor.Dimensions - 1;
        }

        if (dim >= tensor.Dimensions)
        {
            throw new ArgumentOutOfRangeException(nameof(dim));
        }

        int dimSize = tensor.Shape[dim];
        int resultSize = tensor.Size / dimSize;

        var resultData = new float[resultSize];
        var resultShape = new int[tensor.Dimensions - 1];

        // Compute result shape
        int shapeIdx = 0;
        for (int i = 0; i < tensor.Dimensions; i++)
        {
            if (i != dim)
            {
                resultShape[shapeIdx++] = tensor.Shape[i];
            }
        }

        // Compute sum along dimension
        for (int i = 0; i < resultSize; i++)
        {
            float sumVal = 0.0f;

            for (int j = 0; j < dimSize; j++)
            {
                int srcIdx = ComputeIndexForReduce(tensor.Shape, dim, i, j);
                sumVal += tensor.Data[srcIdx];
            }

            resultData[i] = sumVal;
        }

        // Always return with keepDim for easier broadcasting
        var keepDimShape = new int[tensor.Dimensions];
        for (int i = 0; i < tensor.Dimensions; i++)
        {
            keepDimShape[i] = (i == dim) ? 1 : tensor.Shape[i];
        }
        return new Tensor(resultData, keepDimShape, tensor.RequiresGrad, tensor.Dtype);
    }

    /// <summary>
    /// Removes a dimension of size 1 from the tensor.
    /// </summary>
    private static Tensor SqueezeDim(Tensor tensor, int dim)
    {
        if (dim < 0)
        {
            dim = tensor.Dimensions - 1;
        }

        if (dim >= tensor.Dimensions || tensor.Shape[dim] != 1)
        {
            return tensor; // Cannot squeeze, return as-is
        }

        var resultShape = new int[tensor.Dimensions - 1];
        int shapeIdx = 0;
        for (int i = 0; i < tensor.Dimensions; i++)
        {
            if (i != dim)
            {
                resultShape[shapeIdx++] = tensor.Shape[i];
            }
        }

        // Data is the same, just shape changes
        return new Tensor(tensor.Data, resultShape, tensor.RequiresGrad, tensor.Dtype);
    }

    /// <summary>
    /// Computes flat index for reduction operations.
    /// </summary>
    private static int ComputeIndexForReduce(int[] shape, int reduceDim, int outerIdx, int innerIdx)
    {
        var strides = ComputeStrides(shape);

        // Convert outerIdx to indices in all dimensions except reduceDim
        var indices = new int[shape.Length];
        int temp = outerIdx;

        for (int i = 0; i < shape.Length; i++)
        {
            if (i == reduceDim) continue;

            indices[i] = temp % shape[i];
            temp /= shape[i];
        }

        // Set the reduced dimension
        indices[reduceDim] = innerIdx;

        // Compute flat index
        return FlattenIndex(indices, strides);
    }

    /// <summary>
    /// Computes strides for a given shape.
    /// </summary>
    private static int[] ComputeStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        var stride = 1;

        for (var i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }

        return strides;
    }

    /// <summary>
    /// Flattens multi-dimensional indices to a flat index.
    /// </summary>
    private static int FlattenIndex(int[] indices, int[] strides)
    {
        int idx = 0;
        for (int i = 0; i < indices.Length; i++)
        {
            idx += indices[i] * strides[i];
        }
        return idx;
    }
}
