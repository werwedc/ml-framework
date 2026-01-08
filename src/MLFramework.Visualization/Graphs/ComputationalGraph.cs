namespace MLFramework.Visualization.Graphs;

/// <summary>
/// Represents a computational graph with nodes and edges.
/// </summary>
public class ComputationalGraph
{
    /// <summary>
    /// Name of the graph.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Timestamp when the graph was created.
    /// </summary>
    public DateTime Timestamp { get; }

    /// <summary>
    /// Step number (e.g., training step).
    /// </summary>
    public long Step { get; }

    /// <summary>
    /// Dictionary of all nodes in the graph keyed by ID.
    /// </summary>
    public Dictionary<string, GraphNode> Nodes { get; }

    /// <summary>
    /// List of edges in the graph (from node ID to node ID).
    /// </summary>
    public List<(string from, string to)> Edges { get; }

    /// <summary>
    /// Total number of nodes in the graph.
    /// </summary>
    public int NodeCount => Nodes.Count;

    /// <summary>
    /// Total number of edges in the graph.
    /// </summary>
    public int EdgeCount => Edges.Count;

    /// <summary>
    /// Depth of the graph (longest path).
    /// </summary>
    public int Depth { get; private set; }

    /// <summary>
    /// Number of input nodes (nodes with no incoming edges).
    /// </summary>
    public int InputCount { get; private set; }

    /// <summary>
    /// Number of output nodes (nodes with no outgoing edges).
    /// </summary>
    public int OutputCount { get; private set; }

    private readonly HashSet<(string, string)> _edgeSet;

    /// <summary>
    /// Creates a new ComputationalGraph instance.
    /// </summary>
    /// <param name="name">Name of the graph.</param>
    /// <param name="step">Step number.</param>
    public ComputationalGraph(string name, long step = 0)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Step = step;
        Timestamp = DateTime.UtcNow;
        Nodes = new Dictionary<string, GraphNode>();
        Edges = new List<(string, string)>();
        _edgeSet = new HashSet<(string, string)>();
        Depth = 0;
        InputCount = 0;
        OutputCount = 0;
    }

    /// <summary>
    /// Adds a node to the graph.
    /// </summary>
    /// <param name="node">Node to add.</param>
    /// <exception cref="ArgumentNullException">Thrown if node is null.</exception>
    /// <exception cref="ArgumentException">Thrown if a node with the same ID already exists.</exception>
    public void AddNode(GraphNode node)
    {
        if (node == null) throw new ArgumentNullException(nameof(node));
        if (Nodes.ContainsKey(node.Id))
        {
            throw new ArgumentException($"Node with ID '{node.Id}' already exists in the graph.");
        }

        Nodes[node.Id] = node;
        RecalculateStatistics();
    }

    /// <summary>
    /// Adds an edge between two nodes.
    /// </summary>
    /// <param name="fromId">ID of the source node.</param>
    /// <param name="toId">ID of the target node.</param>
    /// <exception cref="ArgumentException">Thrown if edge already exists or nodes don't exist.</exception>
    public void AddEdge(string fromId, string toId)
    {
        if (string.IsNullOrEmpty(fromId))
            throw new ArgumentException("Source node ID cannot be null or empty.", nameof(fromId));
        if (string.IsNullOrEmpty(toId))
            throw new ArgumentException("Target node ID cannot be null or empty.", nameof(toId));
        if (!Nodes.ContainsKey(fromId))
            throw new ArgumentException($"Source node with ID '{fromId}' not found in graph.");
        if (!Nodes.ContainsKey(toId))
            throw new ArgumentException($"Target node with ID '{toId}' not found in graph.");

        var edge = (fromId, toId);
        if (_edgeSet.Contains(edge))
        {
            throw new ArgumentException($"Edge from '{fromId}' to '{toId}' already exists.");
        }

        Edges.Add(edge);
        _edgeSet.Add(edge);

        // Update node connections
        Nodes[fromId].OutputIds.Add(toId);
        Nodes[toId].InputIds.Add(fromId);

        RecalculateStatistics();
    }

    /// <summary>
    /// Gets all input nodes (nodes with no incoming edges).
    /// </summary>
    public IEnumerable<GraphNode> GetInputs()
    {
        return Nodes.Values.Where(n => n.InputIds.Count == 0);
    }

    /// <summary>
    /// Gets all output nodes (nodes with no outgoing edges).
    /// </summary>
    public IEnumerable<GraphNode> GetOutputs()
    {
        return Nodes.Values.Where(n => n.OutputIds.Count == 0);
    }

    /// <summary>
    /// Gets nodes in topological order using Kahn's algorithm.
    /// </summary>
    /// <returns>Nodes in topological order.</returns>
    /// <exception cref="InvalidOperationException">Thrown if the graph contains a cycle.</exception>
    public IEnumerable<GraphNode> GetTopologicalOrder()
    {
        // Calculate in-degrees
        var inDegree = new Dictionary<string, int>();
        foreach (var node in Nodes.Values)
        {
            inDegree[node.Id] = node.InputIds.Count;
        }

        // Initialize queue with nodes that have no incoming edges
        var queue = new Queue<string>();
        foreach (var nodeId in inDegree.Where(kvp => kvp.Value == 0).Select(kvp => kvp.Key))
        {
            queue.Enqueue(nodeId);
        }

        var result = new List<string>();

        while (queue.Count > 0)
        {
            var currentId = queue.Dequeue();
            result.Add(currentId);

            // Decrease in-degree for all neighbors
            foreach (var outputId in Nodes[currentId].OutputIds)
            {
                inDegree[outputId]--;
                if (inDegree[outputId] == 0)
                {
                    queue.Enqueue(outputId);
                }
            }
        }

        // Check for cycles
        if (result.Count != Nodes.Count)
        {
            throw new InvalidOperationException("Graph contains a cycle and cannot be topologically sorted.");
        }

        return result.Select(id => Nodes[id]);
    }

    /// <summary>
    /// Checks if the graph contains a cycle.
    /// </summary>
    public bool HasCycle()
    {
        var visited = new HashSet<string>();
        var recStack = new HashSet<string>();

        bool HasCycleHelper(string nodeId)
        {
            visited.Add(nodeId);
            recStack.Add(nodeId);

            foreach (var neighborId in Nodes[nodeId].OutputIds)
            {
                if (!visited.Contains(neighborId))
                {
                    if (HasCycleHelper(neighborId))
                        return true;
                }
                else if (recStack.Contains(neighborId))
                {
                    return true;
                }
            }

            recStack.Remove(nodeId);
            return false;
        }

        foreach (var nodeId in Nodes.Keys)
        {
            if (!visited.Contains(nodeId))
            {
                if (HasCycleHelper(nodeId))
                    return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Gets disconnected components in the graph.
    /// </summary>
    public List<List<GraphNode>> GetDisconnectedComponents()
    {
        var visited = new HashSet<string>();
        var components = new List<List<GraphNode>>();

        foreach (var nodeId in Nodes.Keys)
        {
            if (!visited.Contains(nodeId))
            {
                var component = new List<GraphNode>();
                DFS(nodeId, visited, component);
                components.Add(component);
            }
        }

        return components;
    }

    /// <summary>
    /// Calculates the maximum depth (longest path) of the graph.
    /// </summary>
    private void CalculateDepth()
    {
        var depth = new Dictionary<string, int>();

        int GetDepth(string nodeId)
        {
            if (depth.ContainsKey(nodeId))
                return depth[nodeId];

            var node = Nodes[nodeId];
            if (node.OutputIds.Count == 0)
            {
                depth[nodeId] = 0;
                return 0;
            }

            int maxChildDepth = 0;
            foreach (var childId in node.OutputIds)
            {
                maxChildDepth = Math.Max(maxChildDepth, GetDepth(childId));
            }

            depth[nodeId] = maxChildDepth + 1;
            return depth[nodeId];
        }

        Depth = 0;
        foreach (var nodeId in Nodes.Keys)
        {
            Depth = Math.Max(Depth, GetDepth(nodeId));
        }
    }

    private void RecalculateStatistics()
    {
        InputCount = GetInputs().Count();
        OutputCount = GetOutputs().Count();
        CalculateDepth();
    }

    private void DFS(string nodeId, HashSet<string> visited, List<GraphNode> component)
    {
        visited.Add(nodeId);
        component.Add(Nodes[nodeId]);

        foreach (var neighborId in Nodes[nodeId].OutputIds)
        {
            if (!visited.Contains(neighborId))
            {
                DFS(neighborId, visited, component);
            }
        }

        // Also traverse backward through inputs
        foreach (var inputId in Nodes[nodeId].InputIds)
        {
            if (!visited.Contains(inputId))
            {
                DFS(inputId, visited, component);
            }
        }
    }
}
