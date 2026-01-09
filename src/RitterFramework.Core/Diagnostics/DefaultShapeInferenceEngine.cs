namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Default implementation of IShapeInferenceEngine with shape inference logic for common operations.
/// </summary>
public class DefaultShapeInferenceEngine : IShapeInferenceEngine
{
    private readonly IOperationMetadataRegistry _registry;

    public DefaultShapeInferenceEngine(IOperationMetadataRegistry registry)
    {
        _registry = registry;
    }

    /// <inheritdoc/>
    public long[] InferOutputShape(
        OperationType operationType,
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters = null)
    {
        var shapes = inputShapes.ToArray();

        switch (operationType)
        {
            case OperationType.MatrixMultiply:
                return InferMatrixMultiplyShape(shapes, operationParameters);

            case OperationType.Conv2D:
                return InferConv2DShape(shapes, operationParameters);

            case OperationType.Linear:
                return InferLinearShape(shapes, operationParameters);

            case OperationType.Concat:
                return InferConcatShape(shapes, operationParameters);

            case OperationType.Stack:
                return InferStackShape(shapes, operationParameters);

            case OperationType.Transpose:
                return InferTransposeShape(shapes, operationParameters);

            case OperationType.Reshape:
                return InferReshapeShape(shapes, operationParameters);

            case OperationType.Broadcast:
                return InferBroadcastShape(shapes);

            default:
                // Fallback: return first input shape
                return shapes.Length > 0 ? shapes[0] : Array.Empty<long>();
        }
    }

    private long[] InferMatrixMultiplyShape(long[][] inputShapes, IDictionary<string, object> parameters)
    {
        // Input: [batch, m] × [m, n] → Output: [batch, n]
        // Input: [m, k] × [k, n] → Output: [m, n]
        var shapeA = inputShapes[0];
        var shapeB = inputShapes[1];

        // Handle batch dimension
        if (shapeA.Length == 3)
        {
            // [batch, m, k] × [k, n] → [batch, m, n]
            return new long[] { shapeA[0], shapeA[1], shapeB[1] };
        }
        else if (shapeA.Length == 2)
        {
            // [m, k] × [k, n] → [m, n]
            return new long[] { shapeA[0], shapeB[1] };
        }
        else
        {
            return shapeA;
        }
    }

    private long[] InferConv2DShape(long[][] inputShapes, IDictionary<string, object> parameters)
    {
        var inputShape = inputShapes[0]; // [N, C_in, H, W]
        var weightShape = inputShapes[1]; // [C_out, C_in, kH, kW]

        long kernelHeight = weightShape[2];
        long kernelWidth = weightShape[3];

        long strideH = 1;
        long strideW = 1;
        long paddingH = 0;
        long paddingW = 0;

        if (parameters != null)
        {
            if (parameters.TryGetValue("stride", out var s) && s is int[] strideArr)
            {
                strideH = strideArr[0];
                strideW = strideArr.Length > 1 ? strideArr[1] : strideH;
            }

            if (parameters.TryGetValue("padding", out var p) && p is int[] paddingArr)
            {
                paddingH = paddingArr[0];
                paddingW = paddingArr.Length > 1 ? paddingArr[1] : paddingH;
            }
        }

        long outputHeight = ((inputShape[2] + 2 * paddingH - kernelHeight) / strideH) + 1;
        long outputWidth = ((inputShape[3] + 2 * paddingW - kernelWidth) / strideW) + 1;

        return new long[] { inputShape[0], weightShape[0], outputHeight, outputWidth };
    }

    private long[] InferLinearShape(long[][] inputShapes, IDictionary<string, object> parameters)
    {
        // Linear is essentially matrix multiply
        var shapeA = inputShapes[0]; // [batch, in_features]
        var weightShape = inputShapes[1]; // [out_features, in_features]

        if (shapeA.Length == 2)
        {
            return new long[] { shapeA[0], weightShape[0] };
        }

        return shapeA;
    }

    private long[] InferConcatShape(long[][] inputShapes, IDictionary<string, object> parameters)
    {
        int axis = 0;
        if (parameters != null && parameters.TryGetValue("axis", out var a))
        {
            axis = Convert.ToInt32(a);
        }

        // All inputs should have same shape except on the concatenation axis
        var firstShape = inputShapes[0];
        long dimSum = 0;

        foreach (var shape in inputShapes)
        {
            dimSum += shape[axis];
        }

        var result = (long[])firstShape.Clone();
        result[axis] = dimSum;
        return result;
    }

    private long[] InferStackShape(long[][] inputShapes, IDictionary<string, object> parameters)
    {
        int axis = 0;
        if (parameters != null && parameters.TryGetValue("axis", out var a))
        {
            axis = Convert.ToInt32(a);
        }

        // All inputs should have same shape
        var firstShape = inputShapes[0];
        int dim = inputShapes.Length;

        var result = new List<long>(firstShape);
        result.Insert(axis, dim);
        return result.ToArray();
    }

    private long[] InferTransposeShape(long[][] inputShapes, IDictionary<string, object> parameters)
    {
        var shape = inputShapes[0];

        if (shape.Length == 2)
        {
            return new long[] { shape[1], shape[0] };
        }

        // For higher dimensions, just reverse the shape (simplified)
        return shape.Reverse().ToArray();
    }

    private long[] InferReshapeShape(long[][] inputShapes, IDictionary<string, object> parameters)
    {
        var inputShape = inputShapes[0];

        if (parameters == null || !parameters.TryGetValue("shape", out var shapeObj))
        {
            return inputShape;
        }

        var targetShape = (long[])shapeObj;

        // Handle -1 (infer dimension)
        long totalElements = 1;
        foreach (var dim in inputShape)
        {
            totalElements *= dim;
        }

        long inferredDim = -1;
        long knownSize = 1;

        for (int i = 0; i < targetShape.Length; i++)
        {
            if (targetShape[i] == -1)
            {
                if (inferredDim != -1)
                {
                    throw new ArgumentException("Only one dimension can be -1 in reshape");
                }
                inferredDim = i;
            }
            else
            {
                knownSize *= targetShape[i];
            }
        }

        var outputShape = (long[])targetShape.Clone();
        if (inferredDim != -1)
        {
            outputShape[inferredDim] = totalElements / knownSize;
        }

        return outputShape;
    }

    private long[] InferBroadcastShape(long[][] inputShapes)
    {
        if (inputShapes.Length == 0)
        {
            return Array.Empty<long>();
        }

        // Start with the shape of the first input
        var result = (long[])inputShapes[0].Clone();

        // Broadcast each subsequent input
        for (int i = 1; i < inputShapes.Length; i++)
        {
            var currentShape = inputShapes[i];

            // Align shapes from the right
            int maxRank = Math.Max(result.Length, currentShape.Length);
            var newResult = new long[maxRank];
            var newCurrent = new long[maxRank];

            Array.Copy(result, 0, newResult, maxRank - result.Length, result.Length);
            Array.Copy(currentShape, 0, newCurrent, maxRank - currentShape.Length, currentShape.Length);

            for (int j = 0; j < maxRank; j++)
            {
                if (newResult[j] == 1)
                {
                    newResult[j] = newCurrent[j];
                }
                else if (newCurrent[j] != 1 && newCurrent[j] != newResult[j])
                {
                    // Cannot broadcast
                    return newResult; // Return partial result
                }
            }

            result = newResult;
        }

        return result;
    }
}
