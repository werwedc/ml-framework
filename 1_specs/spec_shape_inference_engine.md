# Technical Spec: Shape Inference Engine

## Overview
Create a shape inference engine that can predict output tensor shapes for operations without actually executing them. This enables proactive shape checking and better error messages.

## Requirements

### ShapeInferenceEngine Interface
```csharp
public interface IShapeInferenceEngine
{
    // Infer output shape for an operation
    long[] InferOutputShape(
        OperationType operationType,
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters = null);

    // Infer all intermediate shapes in a computation graph
    IDictionary<string, long[]> InferGraphShapes(
        ComputationGraph graph,
        IDictionary<string, long[]> inputShapes);

    // Validate that shapes are compatible with operation
    bool ValidateOperation(
        OperationType operationType,
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters = null,
        out string errorMessage);
}
```

### ComputationGraph Representation
```csharp
public class ComputationGraph
{
    // Map from node name to operation node
    public Dictionary<string, OperationNode> Nodes { get; set; }

    // Edges: from_node -> to_node
    public List<(string from, string to)> Edges { get; set; }
}

public class OperationNode
{
    public string Name { get; set; }
    public OperationType OperationType { get; set; }
    public string[] InputNames { get; set; }
    public IDictionary<string, object> Parameters { get; set; }
}
```

### Default Implementation
Create `DefaultShapeInferenceEngine` implementing `IShapeInferenceEngine`:

#### Matrix Multiply Shape Inference
```csharp
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
    else
    {
        // [m, k] × [k, n] → [m, n]
        return new long[] { shapeA[0], shapeB[1] };
    }
}
```

#### Conv2D Shape Inference
```csharp
private long[] InferConv2DShape(long[][] inputShapes, IDictionary<string, object> parameters)
{
    var inputShape = inputShapes[0]; // [N, C_in, H, W]
    var weightShape = inputShapes[1]; // [C_out, C_in, kH, kW]

    int kernelHeight = (int)weightShape[2];
    int kernelWidth = (int)weightShape[3];
    int strideH = parameters.TryGetValue("stride", out var s) ? ((int[])s)[0] : 1;
    int strideW = parameters.TryGetValue("stride", out var s2) ? ((int[])s2)[1] : 1;
    int paddingH = parameters.TryGetValue("padding", out var p) ? ((int[])p)[0] : 0;
    int paddingW = parameters.TryGetValue("padding", out var p2) ? ((int[])p2)[1] : 0;

    int outputHeight = (int)((inputShape[2] + 2 * paddingH - kernelHeight) / strideH) + 1;
    int outputWidth = (int)((inputShape[3] + 2 * paddingW - kernelWidth) / strideW) + 1;

    return new long[] { inputShape[0], weightShape[0], outputHeight, outputWidth };
}
```

#### Pooling Shape Inference
```csharp
private long[] InferMaxPool2DShape(long[][] inputShapes, IDictionary<string, object> parameters)
{
    var inputShape = inputShapes[0]; // [N, C, H, W]

    int kernelSize = parameters.TryGetValue("kernel_size", out var ks) ? (int)ks : 2;
    int stride = parameters.TryGetValue("stride", out var s) ? (int)s : kernelSize;
    int padding = parameters.TryGetValue("padding", out var p) ? (int)p : 0;

    int outputHeight = (int)((inputShape[2] + 2 * padding - kernelSize) / stride) + 1;
    int outputWidth = (int)((inputShape[3] + 2 * padding - kernelSize) / stride) + 1;

    return new long[] { inputShape[0], inputShape[1], outputHeight, outputWidth };
}
```

#### Flatten Shape Inference
```csharp
private long[] InferFlattenShape(long[][] inputShapes, IDictionary<string, object> parameters)
{
    var inputShape = inputShapes[0];

    int startDim = parameters.TryGetValue("start_dim", out var sd) ? (int)sd : 1;
    int endDim = parameters.TryGetValue("end_dim", out var ed) ? (int)ed : inputShape.Length - 1;

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
```

#### Reshape Shape Inference
```csharp
private long[] InferReshapeShape(long[][] inputShapes, IDictionary<string, object> parameters)
{
    var inputShape = inputShapes[0];
    var targetShape = (long[])parameters["shape"];

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
```

### Graph Shape Inference
```csharp
public IDictionary<string, long[]> InferGraphShapes(
    ComputationGraph graph,
    IDictionary<string, long[]> inputShapes)
{
    var result = new Dictionary<string, long[]>(inputShapes);

    // Topological sort of nodes
    var sortedNodes = TopologicalSort(graph);

    foreach (var nodeName in sortedNodes)
    {
        var node = graph.Nodes[nodeName];

        // Get input shapes from previous nodes
        var inputShapesForOp = node.InputNames
            .Select(name => result[name])
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

    return result;
}
```

## Deliverables
- File: `src/Diagnostics/IShapeInferenceEngine.cs`
- File: `src/Diagnostics/ComputationGraph.cs`
- File: `src/Diagnostics/OperationNode.cs`
- File: `src/Diagnostics/DefaultShapeInferenceEngine.cs`

## Testing Requirements
Create unit tests in `tests/Diagnostics/ShapeInferenceEngineTests.cs`:
- Test MatrixMultiply shape inference (various dimensions)
- Test Conv2D shape inference (different kernel sizes, strides, padding)
- Test MaxPool2D shape inference
- Test AveragePool2D shape inference
- Test Flatten shape inference (different start/end dimensions)
- Test Reshape shape inference (with and without -1)
- Test Concat shape inference
- Test Stack shape inference
- Test graph shape inference on simple linear model
- Test graph shape inference on CNN model
- Test graph shape inference with invalid shapes

## Notes
- This engine should be independent of actual tensor operations
- Use -1 for unknown dimensions in some cases
- Implement fallback for unsupported operations
- Consider caching inference results for performance
- Topological sort should handle cycles gracefully (throw exception)
