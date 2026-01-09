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

            case OperationType.Conv1D:
                return InferConv1DShape(shapes, operationParameters);

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

            case OperationType.Flatten:
                return InferFlattenShape(shapes, operationParameters);

            case OperationType.MaxPool2D:
            case OperationType.AveragePool2D:
                return InferPoolingShape(shapes, operationParameters);

            case OperationType.Broadcast:
                return InferBroadcastShape(shapes);

            default:
                // Fallback: return first input shape
                return shapes.Length > 0 ? shapes[0] : Array.Empty<long>();
        }
    }

    /// <inheritdoc/>
    public IDictionary<string, long[]> InferGraphShapes(
        ComputationGraph graph,
        IDictionary<string, long[]> inputShapes)
    {
        if (graph == null)
        {
            throw new ArgumentNullException(nameof(graph));
        }

        if (inputShapes == null)
        {
            throw new ArgumentNullException(nameof(inputShapes));
        }

        var result = new Dictionary<string, long[]>(inputShapes);

        // Topological sort of nodes
        var sortedNodes = TopologicalSort(graph);

        foreach (var nodeName in sortedNodes)
        {
            var node = graph.Nodes[nodeName];

            // Skip if it's an input node (already in result)
            if (result.ContainsKey(nodeName))
            {
                continue;
            }

            // Get input shapes from previous nodes
            var inputShapesForOp = node.InputNames
                .Select(name => result.ContainsKey(name) ? result[name] : Array.Empty<long>())
                .ToArray();

            // Infer output shape
            var outputShape = InferOutputShape(
                node.OperationType,
                inputShapesForOp,
                node.Parameters);

            result[nodeName] = outputShape;
        }

        return result;
    }

    /// <inheritdoc/>
    public bool ValidateOperation(
        OperationType operationType,
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters,
        out string errorMessage)
    {
        errorMessage = null;

        var shapes = inputShapes.ToArray();

        if (shapes.Length == 0)
        {
            errorMessage = "No input shapes provided";
            return false;
        }

        try
        {
            // Try to infer output shape - this validates input shapes
            var outputShape = InferOutputShape(operationType, shapes, operationParameters);

            // Check for invalid dimensions (zero or negative)
            foreach (var dim in outputShape)
            {
                if (dim <= 0)
                {
                    errorMessage = $"Operation would produce invalid output shape with non-positive dimension: {string.Join(", ", outputShape)}";
                    return false;
                }
            }

            return true;
        }
        catch (Exception ex)
        {
            errorMessage = ex.Message;
            return false;
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

    private long[] InferConv1DShape(long[][] inputShapes, IDictionary<string, object> parameters)
    {
        var inputShape = inputShapes[0]; // [N, C_in, L]
        var weightShape = inputShapes[1]; // [C_out, C_in, k]

        long kernelSize = weightShape[2];
        int stride = parameters?.TryGetValue("stride", out var s) == true ? (int)s : 1;
        int padding = parameters?.TryGetValue("padding", out var p) == true ? (int)p : 0;

        long inputLength = inputShape[2];
        long outputLength = (inputLength + 2 * padding - kernelSize) / stride + 1;

        return new long[] { inputShape[0], weightShape[0], outputLength };
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

    private long[] InferFlattenShape(long[][] inputShapes, IDictionary<string, object> parameters)
    {
        var inputShape = inputShapes[0];

        int startDim = parameters?.TryGetValue("start_dim", out var sd) == true ? (int)sd : 1;
        int endDim = parameters?.TryGetValue("end_dim", out var ed) == true ? (int)ed : inputShape.Length - 1;

        // Calculate total size of flattened dimensions
        long flattenedSize = 1;
        for (int i = startDim; i <= endDim; i++)
        {
            flattenedSize *= inputShape[i];
        }

        // Construct output shape
        var outputShape = new List<long>();
        for (int i = 0; i < startDim; i++)
        {
            outputShape.Add(inputShape[i]);
        }
        outputShape.Add(flattenedSize);

        return outputShape.ToArray();
    }

    private long[] InferPoolingShape(long[][] inputShapes, IDictionary<string, object> parameters)
    {
        var inputShape = inputShapes[0]; // [N, C, H, W]

        int kernelSize = parameters?.TryGetValue("kernel_size", out var ks) == true ? (int)ks : 2;
        int stride = parameters?.TryGetValue("stride", out var s) == true ? (int)s : kernelSize;
        int padding = parameters?.TryGetValue("padding", out var p) == true ? (int)p : 0;

        int inputHeight = (int)inputShape[2];
        int inputWidth = (int)inputShape[3];

        int outputHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
        int outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;

        return new long[] { inputShape[0], inputShape[1], outputHeight, outputWidth };
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

    private List<string> TopologicalSort(ComputationGraph graph)
    {
        // Standard topological sort using Kahn's algorithm
        var result = new List<string>();
        var inDegree = new Dictionary<string, int>();
        var queue = new Queue<string>();

        // Initialize in-degrees
        foreach (var node in graph.Nodes)
        {
            inDegree[node.Key] = 0;
        }

        foreach (var edge in graph.Edges)
        {
            inDegree[edge.to]++;
        }

        // Enqueue nodes with in-degree 0
        foreach (var node in graph.Nodes)
        {
            if (inDegree[node.Key] == 0)
            {
                queue.Enqueue(node.Key);
            }
        }

        // Process nodes
        while (queue.Count > 0)
        {
            var current = queue.Dequeue();
            result.Add(current);

            foreach (var edge in graph.Edges.Where(e => e.from == current))
            {
                inDegree[edge.to]--;
                if (inDegree[edge.to] == 0)
                {
                    queue.Enqueue(edge.to);
                }
            }
        }

        // Check for cycles
        if (result.Count != graph.Nodes.Count)
        {
            throw new InvalidOperationException("Computation graph contains a cycle");
        }

        return result;
    }
}
