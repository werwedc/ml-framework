namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Represents an operation node in a computation graph.
/// Contains operation type, input names, and operation parameters.
/// </summary>
public class OperationNode
{
    /// <summary>
    /// Name of the node (should be unique in the graph).
    /// </summary>
    public string Name { get; set; }

    /// <summary>
    /// The type of operation this node represents.
    /// </summary>
    public OperationType OperationType { get; set; }

    /// <summary>
    /// Names of input nodes that feed into this operation.
    /// </summary>
    public string[] InputNames { get; set; }

    /// <summary>
    /// Operation-specific parameters.
    /// </summary>
    public IDictionary<string, object> Parameters { get; set; }

    /// <summary>
    /// Create a new operation node.
    /// </summary>
    public OperationNode()
    {
        InputNames = Array.Empty<string>();
        Parameters = new Dictionary<string, object>();
    }

    /// <summary>
    /// Create a new operation node with the specified properties.
    /// </summary>
    /// <param name="name">Node name.</param>
    /// <param name="operationType">Operation type.</param>
    /// <param name="inputNames">Input node names.</param>
    /// <param name="parameters">Operation parameters.</param>
    public OperationNode(
        string name,
        OperationType operationType,
        string[] inputNames = null,
        IDictionary<string, object> parameters = null)
    {
        Name = name;
        OperationType = operationType;
        InputNames = inputNames ?? Array.Empty<string>();
        Parameters = parameters ?? new Dictionary<string, object>();
    }

    /// <summary>
    /// Get the number of inputs for this operation.
    /// </summary>
    public int InputCount => InputNames?.Length ?? 0;

    /// <summary>
    /// Get a string representation of this node.
    /// </summary>
    public override string ToString()
    {
        var inputs = InputNames.Length > 0 ? string.Join(", ", InputNames) : "none";
        var paramsStr = Parameters.Count > 0 ? $" params: {Parameters.Count}" : "";
        return $"{Name} ({OperationType}) inputs: [{inputs}]{paramsStr}";
    }
}
