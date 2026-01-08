namespace MLFramework.Visualization.Graphs;

/// <summary>
/// Contains analysis results for a computational graph.
/// </summary>
public class GraphAnalysis
{
    /// <summary>
    /// Total number of parameters in the graph.
    /// </summary>
    public int TotalParameters { get; }

    /// <summary>
    /// Total number of operations in the graph.
    /// </summary>
    public int TotalOperations { get; }

    /// <summary>
    /// Count of operations grouped by type.
    /// </summary>
    public Dictionary<string, int> OperationCounts { get; }

    /// <summary>
    /// Depth of the graph.
    /// </summary>
    public int GraphDepth { get; }

    /// <summary>
    /// Maximum fan-in (number of incoming edges) in the graph.
    /// </summary>
    public int MaxFanIn { get; }

    /// <summary>
    /// Maximum fan-out (number of outgoing edges) in the graph.
    /// </summary>
    public int MaxFanOut { get; }

    /// <summary>
    /// List of warnings about potential issues in the graph.
    /// </summary>
    public List<string> Warnings { get; }

    /// <summary>
    /// List of recommendations for improving the graph.
    /// </summary>
    public List<string> Recommendations { get; }

    /// <summary>
    /// Creates a new GraphAnalysis instance.
    /// </summary>
    public GraphAnalysis(
        int totalParameters,
        int totalOperations,
        Dictionary<string, int> operationCounts,
        int graphDepth,
        int maxFanIn,
        int maxFanOut,
        List<string>? warnings = null,
        List<string>? recommendations = null)
    {
        TotalParameters = totalParameters;
        TotalOperations = totalOperations;
        OperationCounts = operationCounts ?? new Dictionary<string, int>();
        GraphDepth = graphDepth;
        MaxFanIn = maxFanIn;
        MaxFanOut = maxFanOut;
        Warnings = warnings ?? new List<string>();
        Recommendations = recommendations ?? new List<string>();
    }
}

/// <summary>
/// Analyzes computational graphs and provides insights.
/// </summary>
public class GraphAnalyzer
{
    /// <summary>
    /// Analyzes a computational graph and returns detailed analysis.
    /// </summary>
    /// <param name="graph">The graph to analyze.</param>
    /// <returns>Graph analysis results.</returns>
    public GraphAnalysis Analyze(ComputationalGraph graph)
    {
        if (graph == null) throw new ArgumentNullException(nameof(graph));

        var warnings = new List<string>();
        var recommendations = new List<string>();
        var operationCounts = new Dictionary<string, int>();

        int totalParameters = 0;
        int totalOperations = 0;
        int maxFanIn = 0;
        int maxFanOut = 0;

        // Count parameters, operations, and calculate fan-in/fan-out
        foreach (var node in graph.Nodes.Values)
        {
            switch (node.Type)
            {
                case NodeType.Parameter:
                    totalParameters++;
                    break;
                case NodeType.Operation:
                    totalOperations++;
                    var opType = node.OpType ?? "Unknown";
                    if (operationCounts.ContainsKey(opType))
                    {
                        operationCounts[opType]++;
                    }
                    else
                    {
                        operationCounts[opType] = 1;
                    }
                    break;
            }

            maxFanIn = Math.Max(maxFanIn, node.InputIds.Count);
            maxFanOut = Math.Max(maxFanOut, node.OutputIds.Count);
        }

        // Check for potential issues
        if (graph.Depth > 100)
        {
            warnings.Add($"Graph is very deep ({graph.Depth} layers). This may cause vanishing/exploding gradients.");
            recommendations.Add("Consider using skip connections or normalization layers.");
        }

        if (maxFanIn > 10)
        {
            warnings.Add($"Some nodes have high fan-in (max: {maxFanIn}). This may be a bottleneck.");
            recommendations.Add("Consider restructuring to reduce fan-in, or verify if this is intentional.");
        }

        if (maxFanOut > 10)
        {
            warnings.Add($"Some nodes have high fan-out (max: {maxFanOut}). This may indicate a branching structure.");
            recommendations.Add("Consider if all branches are necessary or can be consolidated.");
        }

        // Check for skip connections
        int skipConnections = CountSkipConnections(graph);
        if (skipConnections > 0)
        {
            recommendations.Add($"Graph contains {skipConnections} skip connections (good for gradient flow).");
        }

        // Check for residual connections
        if (HasResidualConnections(graph))
        {
            recommendations.Add("Graph appears to use residual connections (good for deep networks).");
        }

        // Check for disconnected components
        var components = graph.GetDisconnectedComponents();
        if (components.Count > 1)
        {
            warnings.Add($"Graph has {components.Count} disconnected components.");
            recommendations.Add("Verify if this is intentional. Disconnected components may indicate missing connections.");
        }

        // Check for cycles
        if (graph.HasCycle())
        {
            warnings.Add("Graph contains cycles. This may cause issues during topological sort.");
            recommendations.Add("Review graph structure and remove intentional cycles if not required.");
        }

        // Check for common patterns
        AnalyzeCommonPatterns(operationCounts, recommendations);

        return new GraphAnalysis(
            totalParameters,
            totalOperations,
            operationCounts,
            graph.Depth,
            maxFanIn,
            maxFanOut,
            warnings,
            recommendations);
    }

    private int CountSkipConnections(ComputationalGraph graph)
    {
        int skipCount = 0;

        foreach (var node in graph.Nodes.Values)
        {
            // Check if any input connects to the same output through multiple paths
            foreach (var inputId in node.InputIds)
            {
                var inputs = new HashSet<string>();
                CollectInputs(graph, inputId, inputs);

                if (inputs.Count > 1)
                {
                    // Potential skip connection detected
                    skipCount++;
                }
            }
        }

        return skipCount;
    }

    private void CollectInputs(ComputationalGraph graph, string nodeId, HashSet<string> visited)
    {
        if (visited.Contains(nodeId)) return;

        visited.Add(nodeId);

        if (!graph.Nodes.TryGetValue(nodeId, out var node)) return;

        foreach (var inputId in node.InputIds)
        {
            CollectInputs(graph, inputId, visited);
        }
    }

    private bool HasResidualConnections(ComputationalGraph graph)
    {
        // Simple heuristic: if there are nodes that have both add operations and multiple inputs from similar depths
        foreach (var node in graph.Nodes.Values)
        {
            if (node.OpType == "Add" || node.OpType == "+")
            {
                if (node.InputIds.Count >= 2)
                {
                    // Check if inputs come from different parts of the network
                    var inputDepths = new HashSet<int>();
                    foreach (var inputId in node.InputIds)
                    {
                        inputDepths.Add(GetNodeDepth(graph, inputId));
                    }

                    if (inputDepths.Count > 1)
                    {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    private int GetNodeDepth(ComputationalGraph graph, string nodeId)
    {
        if (!graph.Nodes.TryGetValue(nodeId, out var node)) return 0;

        if (node.InputIds.Count == 0) return 0;

        int maxInputDepth = 0;
        foreach (var inputId in node.InputIds)
        {
            maxInputDepth = Math.Max(maxInputDepth, GetNodeDepth(graph, inputId));
        }

        return maxInputDepth + 1;
    }

    private void AnalyzeCommonPatterns(Dictionary<string, int> operationCounts, List<string> recommendations)
    {
        // Check for common ML patterns
        if (operationCounts.ContainsKey("Conv2D") && operationCounts.ContainsKey("BatchNorm"))
        {
            recommendations.Add("Graph uses Conv2D + BatchNorm pattern (good practice).");
        }

        if (operationCounts.ContainsKey("Dropout"))
        {
            recommendations.Add("Graph uses Dropout for regularization.");
        }

        if (operationCounts.ContainsKey("MaxPool2D") || operationCounts.ContainsKey("AvgPool2D"))
        {
            recommendations.Add("Graph uses pooling layers for spatial downsampling.");
        }

        if (operationCounts.ContainsKey("ReLU") || operationCounts.ContainsKey("LeakyReLU") ||
            operationCounts.ContainsKey("GELU") || operationCounts.ContainsKey("Swish"))
        {
            recommendations.Add("Graph uses activation functions.");
        }
    }
}
