namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Represents a computation graph for shape inference.
/// Contains nodes representing operations and edges representing data flow.
/// </summary>
public class ComputationGraph
{
    /// <summary>
    /// Map from node name to operation node.
    /// </summary>
    public Dictionary<string, OperationNode> Nodes { get; set; }

    /// <summary>
    /// Edges: from_node -> to_node
    /// </summary>
    public List<(string from, string to)> Edges { get; set; }

    /// <summary>
    /// Create an empty computation graph.
    /// </summary>
    public ComputationGraph()
    {
        Nodes = new Dictionary<string, OperationNode>();
        Edges = new List<(string from, string to)>();
    }

    /// <summary>
    /// Add a node to the graph.
    /// </summary>
    /// <param name="node">The operation node to add.</param>
    public void AddNode(OperationNode node)
    {
        if (node == null)
        {
            throw new ArgumentNullException(nameof(node));
        }

        if (string.IsNullOrEmpty(node.Name))
        {
            throw new ArgumentException("Node name cannot be null or empty", nameof(node));
        }

        Nodes[node.Name] = node;
    }

    /// <summary>
    /// Add an edge from one node to another.
    /// </summary>
    /// <param name="fromNode">Source node name.</param>
    /// <param name="toNode">Target node name.</param>
    public void AddEdge(string fromNode, string toNode)
    {
        if (string.IsNullOrEmpty(fromNode))
        {
            throw new ArgumentException("Source node name cannot be null or empty", nameof(fromNode));
        }

        if (string.IsNullOrEmpty(toNode))
        {
            throw new ArgumentException("Target node name cannot be null or empty", nameof(toNode));
        }

        if (!Nodes.ContainsKey(fromNode))
        {
            throw new ArgumentException($"Source node '{fromNode}' does not exist in the graph", nameof(fromNode));
        }

        if (!Nodes.ContainsKey(toNode))
        {
            throw new ArgumentException($"Target node '{toNode}' does not exist in the graph", nameof(toNode));
        }

        Edges.Add((fromNode, toNode));
    }

    /// <summary>
    /// Get all nodes that have no incoming edges (input nodes).
    /// </summary>
    public List<string> GetInputNodes()
    {
        var inputNodes = new List<string>();

        // Find nodes with in-degree 0
        foreach (var node in Nodes)
        {
            bool hasIncomingEdges = Edges.Any(e => e.to == node.Key);
            if (!hasIncomingEdges)
            {
                inputNodes.Add(node.Key);
            }
        }

        return inputNodes;
    }

    /// <summary>
    /// Get all nodes that have no outgoing edges (output nodes).
    /// </summary>
    public List<string> GetOutputNodes()
    {
        var outputNodes = new List<string>();

        // Find nodes with out-degree 0
        foreach (var node in Nodes)
        {
            bool hasOutgoingEdges = Edges.Any(e => e.from == node.Key);
            if (!hasOutgoingEdges)
            {
                outputNodes.Add(node.Key);
            }
        }

        return outputNodes;
    }
}
