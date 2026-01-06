using RitterFramework.Core.Tensor;
using System.Collections.Concurrent;

namespace MLFramework.Autograd;

/// <summary>
/// Represents a node in the computational graph, tracking tensor dependencies and operation history.
/// </summary>
public class GraphNode : IDisposable
{
    private static int _nextId = 0;
    private readonly object _lock = new object();
    private bool _disposed = false;

    /// <summary>
    /// Gets the output tensor produced by this node's operation.
    /// </summary>
    public Tensor OutputTensor { get; }

    /// <summary>
    /// Gets the child nodes (inputs) that this node depends on.
    /// </summary>
    public IReadOnlyList<GraphNode> Children { get; }

    /// <summary>
    /// Gets the operation context containing metadata and backward function.
    /// </summary>
    public OperationContext Operation { get; }

    /// <summary>
    /// Gets whether this node is a leaf node (no children).
    /// </summary>
    public bool IsLeaf => Children.Count == 0;

    /// <summary>
    /// Gets the unique gradient function identifier for this node.
    /// </summary>
    public int GradFnId { get; }

    /// <summary>
    /// Gets whether this node has been registered with a graph builder.
    /// </summary>
    public bool IsRegistered { get; private set; }

    /// <summary>
    /// Initializes a new instance of the GraphNode class.
    /// </summary>
    /// <param name="output">The output tensor produced by this operation.</param>
    /// <param name="operation">The operation context for this node.</param>
    /// <param name="children">The child nodes (inputs) that this node depends on.</param>
    public GraphNode(Tensor output, OperationContext operation, params GraphNode[] children)
    {
        OutputTensor = output ?? throw new ArgumentNullException(nameof(output));
        Operation = operation ?? throw new ArgumentNullException(nameof(operation));
        Children = children?.ToList() ?? new List<GraphNode>();
        GradFnId = Interlocked.Increment(ref _nextId);
    }

    /// <summary>
    /// Registers this node with the current graph builder.
    /// </summary>
    public void Register()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GraphNode));

        var graph = GraphBuilder.GetCurrent();
        if (graph != null && graph.IsEnabled)
        {
            lock (_lock)
            {
                if (!IsRegistered)
                {
                    graph.CreateNode(OutputTensor, Operation, Children.ToArray());
                    IsRegistered = true;
                }
            }
        }
    }

    /// <summary>
    /// Disposes of the node's resources.
    /// </summary>
    public void Dispose()
    {
        lock (_lock)
        {
            if (!_disposed)
            {
                // Clear references to help GC
                // Note: We don't dispose OutputTensor or Children as they're managed externally
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }
    }

    /// <summary>
    /// Finalizer for GraphNode.
    /// </summary>
    ~GraphNode()
    {
        Dispose();
    }
}
