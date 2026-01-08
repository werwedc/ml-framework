# Spec: Tracing Infrastructure

## Overview
Implement the execution tracing infrastructure that will be used by the JIT compiler. This infrastructure tracks tensor operations to build a computational graph.

## Scope
- Define TracedTensor and TraceNode classes
- Implement operation recording
- Support shape and type inference
- Create trace context management

## Technical Requirements

### 1. TraceNode Class

```csharp
namespace MLFramework.Functional.Tracing
{
    /// <summary>
    /// Represents a single operation in a computational trace.
    /// </summary>
    public class TraceNode
    {
        public Guid Id { get; } = Guid.NewGuid();
        public string OperationName { get; }
        public TraceNode[] Inputs { get; }
        public TensorShape OutputShape { get; }
        public TensorType OutputType { get; }
        public Dictionary<string, object> Attributes { get; }

        public TraceNode(
            string operationName,
            TraceNode[] inputs,
            TensorShape outputShape,
            TensorType outputType,
            Dictionary<string, object> attributes = null)
        {
            OperationName = operationName;
            Inputs = inputs ?? Array.Empty<TraceNode>();
            OutputShape = outputShape;
            OutputType = outputType;
            Attributes = attributes ?? new Dictionary<string, object>();
        }

        public override string ToString()
        {
            return $"{OperationName}({OutputShape})";
        }
    }

    public enum TensorType
    {
        Float32,
        Float64,
        Int32,
        Int64,
        Bool
    }
}
```

### 2. TracedTensor Class

```csharp
/// <summary>
/// A tensor wrapper that records operations for tracing.
/// </summary>
public class TracedTensor
{
    private readonly Tensor _underlying;  // Actual tensor (for eager execution)
    private readonly TraceNode _node;     // Corresponding trace node

    /// <summary>
    /// The underlying actual tensor.
    /// </summary>
    public Tensor Underlying => _underlying;

    /// <summary>
    /// The trace node representing this tensor's computation.
    /// </summary>
    public TraceNode Node => _node;

    /// <summary>
    /// Shape of the tensor.
    /// </summary>
    public TensorShape Shape => _underlying.Shape;

    /// <summary>
    /// Type of the tensor.
    /// </summary>
    public TensorType Type => _underlying.Type;

    private TracedTensor(Tensor underlying, TraceNode node)
    {
        _underlying = underlying ?? throw new ArgumentNullException(nameof(underlying));
        _node = node ?? throw new ArgumentNullException(nameof(node));
    }

    /// <summary>
    /// Create a traced tensor from a regular tensor (input tensor).
    /// </summary>
    public static TracedTensor Create(Tensor tensor, string name = "input")
    {
        var node = new TraceNode(name, Array.Empty<TraceNode>(), tensor.Shape, tensor.Type);
        return new TracedTensor(tensor, node);
    }

    /// <summary>
    /// Create a traced tensor from an operation.
    /// </summary>
    public static TracedTensor Create(Tensor result, string operation, TracedTensor[] inputs, Dictionary<string, object> attributes = null)
    {
        var inputNodes = inputs.Select(t => t.Node).ToArray();
        var node = new TraceNode(operation, inputNodes, result.Shape, result.Type, attributes);
        return new TracedTensor(result, node);
    }

    // Implicit conversion for convenience
    public static implicit operator Tensor(TracedTensor traced) => traced._underlying;

    // Wrapper operations that record to trace
    public TracedTensor Add(TracedTensor other)
    {
        var result = _underlying.Add(other._underlying);
        return Create(result, "add", new[] { this, other });
    }

    public TracedTensor Multiply(TracedTensor other)
    {
        var result = _underlying.Multiply(other._underlying);
        return Create(result, "multiply", new[] { this, other });
    }

    public TracedTensor MatMul(TracedTensor other)
    {
        var result = _underlying.MatMul(other._underlying);
        return Create(result, "matmul", new[] { this, other });
    }

    public TracedTensor ReLU()
    {
        var result = _underlying.ReLU();
        return Create(result, "relu", new[] { this });
    }
}
```

### 3. TraceContext Class

```csharp
/// <summary>
/// Manages the current trace context.
/// </summary>
public class TraceContext : IDisposable
{
    private static readonly ThreadLocal<TraceContext> _current = new ThreadLocal<TraceContext>();

    public static TraceContext Current => _current.Value;

    public bool IsActive => _current.Value == this;
    public List<TraceNode> Nodes { get; } = new List<TraceNode>();
    public Dictionary<string, TraceNode> NamedOutputs { get; } = new Dictionary<string, TraceNode>();

    public TraceContext()
    {
        _current.Value = this;
    }

    public void RecordNode(TraceNode node)
    {
        if (!IsActive)
            throw new InvalidOperationException("Trace context is not active");

        Nodes.Add(node);
    }

    public void RegisterOutput(string name, TraceNode node)
    {
        if (!IsActive)
            throw new InvalidOperationException("Trace context is not active");

        NamedOutputs[name] = node;
    }

    public void Dispose()
    {
        if (IsActive)
            _current.Value = null;
    }

    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine("Trace:");
        foreach (var node in Nodes)
        {
            sb.AppendLine($"  {node}");
        }
        return sb.ToString();
    }
}
```

### 4. TensorShape Helper

```csharp
/// <summary>
/// Immutable tensor shape representation.
/// </summary>
public class TensorShape : IEquatable<TensorShape>
{
    private readonly int[] _dimensions;

    public int Rank => _dimensions.Length;
    public IReadOnlyList<int> Dimensions => _dimensions;

    public int this[int index] => _dimensions[index];

    public TensorShape(params int[] dimensions)
    {
        _dimensions = dimensions ?? Array.Empty<int>();
    }

    public int TotalElements => _dimensions.Aggregate(1, (a, b) => a * b);

    public bool Equals(TensorShape other)
    {
        if (other == null) return false;
        if (Rank != other.Rank) return false;

        for (int i = 0; i < Rank; i++)
        {
            if (_dimensions[i] != other._dimensions[i])
                return false;
        }

        return true;
    }

    public override bool Equals(object obj) => Equals(obj as TensorShape);

    public override int GetHashCode()
    {
        return _dimensions.Aggregate(17, (hash, dim) => hash * 31 + dim.GetHashCode());
    }

    public override string ToString()
    {
        return $"[{string.Join(", ", _dimensions)}]";
    }

    public static TensorShape Scalar => new TensorShape();
}
```

## Files to Create
1. `src/MLFramework/Functional/Tracing/TraceNode.cs`
2. `src/MLFramework/Functional/Tracing/TracedTensor.cs`
3. `src/MLFramework/Functional/Tracing/TraceContext.cs`
4. `src/MLFramework/Functional/Tracing/TensorShape.cs`
5. `src/MLFramework/Functional/Tracing/TensorType.cs`

## Dependencies
- spec_functional_core_interfaces.md
- MLFramework.Tensor with basic operations (Add, Multiply, MatMul, ReLU)

## Success Criteria
- Can create a trace context
- Operations on TracedTensor are recorded
- Trace can be serialized/printed
- Shape and type are preserved through operations
- Multiple trace contexts can be created (disposable)

## Notes for Coder
- This is infrastructure - keep it simple
- TracedTensor wraps real Tensor and records operations
- Only implement a few basic operations (add, mul, matmul, relu)
- Extending to more operations will happen in JIT spec
- Focus on the trace structure, not actual compilation yet
- Use ThreadLocal to support concurrent tracing

## Example Usage
```csharp
using (var trace = new TraceContext())
{
    var x = TracedTensor.Create(new Tensor(new[] { 1, 2, 3 }), "x");
    var y = TracedTensor.Create(new Tensor(new[] { 4, 5, 6 }), "y");

    var result = x.Add(y);
    var final = result.Multiply(x);

    trace.RegisterOutput("final", final.Node);

    Console.WriteLine(trace);
    // Output:
    // Trace:
    //   input([3])
    //   input([3])
    //   add([3])
    //   multiply([3])
}
```
