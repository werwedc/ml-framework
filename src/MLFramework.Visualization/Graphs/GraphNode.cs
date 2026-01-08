using MLFramework.Core;

namespace MLFramework.Visualization.Graphs;

/// <summary>
/// Represents a node in a computational graph.
/// </summary>
public class GraphNode
{
    /// <summary>
    /// Unique identifier for this node.
    /// </summary>
    public string Id { get; }

    /// <summary>
    /// Human-readable name for this node.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// The type of this node.
    /// </summary>
    public NodeType Type { get; }

    /// <summary>
    /// Operation type (e.g., "Conv2D", "ReLU", "MatMul").
    /// </summary>
    public string OpType { get; }

    /// <summary>
    /// Tensor shape information.
    /// </summary>
    public long[] Shape { get; }

    /// <summary>
    /// Data type of the tensor.
    /// </summary>
    public DataType DataType { get; }

    /// <summary>
    /// List of input node IDs.
    /// </summary>
    public List<string> InputIds { get; }

    /// <summary>
    /// List of output node IDs.
    /// </summary>
    public List<string> OutputIds { get; }

    /// <summary>
    /// Control dependencies for this node.
    /// </summary>
    public List<string> ControlDependencies { get; }

    /// <summary>
    /// Operation-specific attributes (e.g., kernel_size, stride).
    /// </summary>
    public Dictionary<string, object> Attributes { get; }

    /// <summary>
    /// Additional metadata for extensibility.
    /// </summary>
    public Dictionary<string, string> Metadata { get; }

    /// <summary>
    /// Creates a new GraphNode instance.
    /// </summary>
    /// <param name="id">Unique identifier for the node.</param>
    /// <param name="name">Human-readable name.</param>
    /// <param name="type">Type of the node.</param>
    /// <param name="opType">Operation type (optional, empty for non-operation nodes).</param>
    /// <param name="shape">Tensor shape (optional).</param>
    /// <param name="dataType">Data type of the tensor.</param>
    /// <param name="inputIds">List of input node IDs (optional).</param>
    /// <param name="outputIds">List of output node IDs (optional).</param>
    /// <param name="controlDependencies">Control dependencies (optional).</param>
    /// <param name="attributes">Operation-specific attributes (optional).</param>
    /// <param name="metadata">Additional metadata (optional).</param>
    public GraphNode(
        string id,
        string name,
        NodeType type,
        string opType = "",
        long[]? shape = null,
        DataType dataType = DataType.Float32,
        List<string>? inputIds = null,
        List<string>? outputIds = null,
        List<string>? controlDependencies = null,
        Dictionary<string, object>? attributes = null,
        Dictionary<string, string>? metadata = null)
    {
        Id = id ?? throw new ArgumentNullException(nameof(id));
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Type = type;
        OpType = opType ?? "";
        Shape = shape ?? Array.Empty<long>();
        DataType = dataType;
        InputIds = inputIds ?? new List<string>();
        OutputIds = outputIds ?? new List<string>();
        ControlDependencies = controlDependencies ?? new List<string>();
        Attributes = attributes ?? new Dictionary<string, object>();
        Metadata = metadata ?? new Dictionary<string, string>();
    }

    /// <summary>
    /// Creates a shallow copy of this node.
    /// </summary>
    public GraphNode Clone()
    {
        return new GraphNode(
            Id,
            Name,
            Type,
            OpType,
            Shape.ToArray(),
            DataType,
            new List<string>(InputIds),
            new List<string>(OutputIds),
            new List<string>(ControlDependencies),
            new Dictionary<string, object>(Attributes),
            new Dictionary<string, string>(Metadata));
    }
}
