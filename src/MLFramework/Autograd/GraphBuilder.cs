using RitterFramework.Core.Tensor;
using System.Collections.Concurrent;

namespace MLFramework.Autograd;

/// <summary>
/// Manages the construction of the computational graph during forward propagation.
/// Thread-local instances ensure thread-safe graph building.
/// </summary>
public class GraphBuilder : IDisposable
{
    // Thread-local storage for graph builder instances
    private static readonly ConcurrentDictionary<int, GraphBuilder> _instances = new();

    private readonly int _threadId;
    private readonly object _lock = new object();
    private readonly Stack<GraphNode> _nodeStack;
    private readonly List<GraphNode> _allNodes;
    private bool _isEnabled = true;
    private bool _disposed = false;

    /// <summary>
    /// Gets or sets whether graph building is enabled.
    /// When disabled, no nodes will be created during forward pass.
    /// </summary>
    public bool IsEnabled
    {
        get => _isEnabled;
        set => _isEnabled = value;
    }

    /// <summary>
    /// Gets the current node being built (top of the stack).
    /// </summary>
    public GraphNode? CurrentNode => _nodeStack.Count > 0 ? _nodeStack.Peek() : null;

    /// <summary>
    /// Gets the node stack for scope management.
    /// </summary>
    public IReadOnlyList<GraphNode> NodeStack => _nodeStack.ToList().AsReadOnly();

    /// <summary>
    /// Gets the total number of nodes in the graph.
    /// </summary>
    public int NodeCount => _allNodes.Count;

    /// <summary>
    /// Initializes a new instance of the GraphBuilder class for the current thread.
    /// </summary>
    public GraphBuilder()
    {
        _threadId = Environment.CurrentManagedThreadId;
        _nodeStack = new Stack<GraphNode>();
        _allNodes = new List<GraphNode>();

        // Register this instance
        _instances.TryAdd(_threadId, this);
    }

    /// <summary>
    /// Gets the current thread's graph builder instance.
    /// </summary>
    /// <returns>The current graph builder, or null if none exists.</returns>
    public static GraphBuilder? GetCurrent()
    {
        var threadId = Environment.CurrentManagedThreadId;
        return _instances.TryGetValue(threadId, out var builder) ? builder : null;
    }

    /// <summary>
    /// Creates a new graph node and adds it to the graph.
    /// </summary>
    /// <param name="output">The output tensor produced by the operation.</param>
    /// <param name="operation">The operation context for this node.</param>
    /// <param name="children">The child nodes (inputs) that this node depends on.</param>
    /// <returns>The created graph node.</returns>
    public GraphNode CreateNode(Tensor output, OperationContext operation, params GraphNode[] children)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GraphBuilder));

        if (!_isEnabled)
            throw new InvalidOperationException("Graph building is disabled");

        var node = new GraphNode(output, operation, children);

        lock (_lock)
        {
            _allNodes.Add(node);
        }

        return node;
    }

    /// <summary>
    /// Pushes a node onto the scope stack.
    /// </summary>
    /// <param name="node">The node to push.</param>
    public void PushScope(GraphNode node)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GraphBuilder));

        if (node == null)
            throw new ArgumentNullException(nameof(node));

        lock (_lock)
        {
            _nodeStack.Push(node);
        }
    }

    /// <summary>
    /// Pops the current node from the scope stack.
    /// </summary>
    /// <returns>The popped node.</returns>
    public GraphNode PopScope()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GraphBuilder));

        lock (_lock)
        {
            if (_nodeStack.Count == 0)
                throw new InvalidOperationException("Cannot pop from empty node stack");

            return _nodeStack.Pop();
        }
    }

    /// <summary>
    /// Clears the entire graph, disposing all nodes.
    /// </summary>
    public void ClearGraph()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GraphBuilder));

        lock (_lock)
        {
            // Clear the node stack
            _nodeStack.Clear();

            // Dispose all nodes
            foreach (var node in _allNodes)
            {
                node.Dispose();
            }

            _allNodes.Clear();
        }
    }

    /// <summary>
    /// Gets all root nodes (nodes with no parents).
    /// </summary>
    /// <returns>A list of root nodes.</returns>
    public List<GraphNode> GetRootNodes()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GraphBuilder));

        lock (_lock)
        {
            // Find nodes that are not children of any other node
            var allChildren = new HashSet<GraphNode>();
            foreach (var node in _allNodes)
            {
                foreach (var child in node.Children)
                {
                    allChildren.Add(child);
                }
            }

            var rootNodes = _allNodes.Where(n => !allChildren.Contains(n)).ToList();
            return rootNodes;
        }
    }

    /// <summary>
    /// Gets all leaf nodes (nodes with no children).
    /// </summary>
    /// <returns>A list of leaf nodes.</returns>
    public List<GraphNode> GetLeafNodes()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(GraphBuilder));

        lock (_lock)
        {
            return _allNodes.Where(n => n.IsLeaf).ToList();
        }
    }

    /// <summary>
    /// Disposes of the graph builder and all its nodes.
    /// </summary>
    public void Dispose()
    {
        lock (_lock)
        {
            if (!_disposed)
            {
                ClearGraph();
                _instances.TryRemove(_threadId, out _);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }
    }

    /// <summary>
    /// Finalizer for GraphBuilder.
    /// </summary>
    ~GraphBuilder()
    {
        Dispose();
    }
}
