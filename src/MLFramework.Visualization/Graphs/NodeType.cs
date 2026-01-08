namespace MLFramework.Visualization.Graphs;

/// <summary>
/// Defines the different types of nodes in a computational graph.
/// </summary>
public enum NodeType
{
    /// <summary>
    /// Mathematical operation node (e.g., Conv2D, MatMul, ReLU)
    /// </summary>
    Operation,

    /// <summary>
    /// Data tensor node representing intermediate or final tensors
    /// </summary>
    Tensor,

    /// <summary>
    /// Model parameter node (weights, biases, etc.)
    /// </summary>
    Parameter,

    /// <summary>
    /// Constant value node
    /// </summary>
    Constant,

    /// <summary>
    /// Input placeholder node for model inputs
    /// </summary>
    Placeholder
}
