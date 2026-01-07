using MLFramework.Core;

namespace MLFramework.Fusion;

/// <summary>
/// Represents an operation in a computational graph
/// </summary>
public abstract record Operation
{
    /// <summary>
    /// Gets the unique identifier for this operation
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Gets the operation type name (e.g., "Add", "Conv2D", "ReLU")
    /// </summary>
    public required string Type { get; init; }

    /// <summary>
    /// Gets the name of this operation instance
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Gets the data type of this operation's tensors
    /// </summary>
    public required DataType DataType { get; init; }

    /// <summary>
    /// Gets the memory layout for this operation
    /// </summary>
    public required TensorLayout Layout { get; init; }

    /// <summary>
    /// Gets the input tensor shape
    /// </summary>
    public required TensorShape InputShape { get; init; }

    /// <summary>
    /// Gets the output tensor shape
    /// </summary>
    public required TensorShape OutputShape { get; init; }

    /// <summary>
    /// Gets the input tensors for this operation
    /// </summary>
    public required IReadOnlyList<string> Inputs { get; init; }

    /// <summary>
    /// Gets the output tensors for this operation
    /// </summary>
    public required IReadOnlyList<string> Outputs { get; init; }

    /// <summary>
    /// Gets the operation attributes
    /// </summary>
    public required IReadOnlyDictionary<string, object> Attributes { get; init; }
}
