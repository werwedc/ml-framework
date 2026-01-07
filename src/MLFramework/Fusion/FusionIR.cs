using MLFramework.Core;

namespace MLFramework.Fusion;

/// <summary>
/// Intermediate representation for fused operations before kernel generation
/// </summary>
public class FusionIR
{
    public required string Id { get; init; }
    public required IReadOnlyList<FusionOpNode> Nodes { get; init; }
    public required IReadOnlyList<FusionVariable> Variables { get; init; }
    public required MemoryLayout MemoryLayout { get; init; }
    public required ComputeRequirements ComputeRequirements { get; init; }

    /// <summary>
    /// Gets the dataflow graph of the fusion
    /// </summary>
    public FusionDataflowGraph BuildDataflowGraph()
    {
        var graph = new FusionDataflowGraph();
        var definedVars = new HashSet<string>();

        // Add all variables as nodes
        foreach (var variable in Variables)
        {
            graph.AddVariable(variable);
        }

        // Add edges based on node connections
        foreach (var node in Nodes)
        {
            graph.AddOperation(node);

            // For each input variable, add edge from variable to this operation
            foreach (var inputVar in node.InputVars)
            {
                if (inputVar != "input" && !definedVars.Contains(inputVar))
                {
                    throw new InvalidOperationException($"Variable {inputVar} used before definition");
                }

                if (Variables.FirstOrDefault(v => v.Name == inputVar) is { } variable)
                {
                    graph.AddEdge(variable, node);
                }
            }

            // Mark output variable as defined
            definedVars.Add(node.OutputVar);

            // Add edge from this operation to output variable
            if (Variables.FirstOrDefault(v => v.Name == node.OutputVar) is { } outputVar)
            {
                graph.AddEdge(node, outputVar);
            }
        }

        return graph;
    }
}

/// <summary>
/// Represents a node in the fusion IR
/// </summary>
public record FusionOpNode
{
    public required string Id { get; init; }
    public required string OriginalOpType { get; init; }
    public required IReadOnlyList<string> InputVars { get; init; }
    public required string OutputVar { get; init; }
    public required Dictionary<string, object> Attributes { get; init; }
}

/// <summary>
/// Represents a variable in the fusion IR
/// </summary>
public record FusionVariable
{
    public required string Name { get; init; }
    public required TensorShape Shape { get; init; }
    public required DataType DataType { get; init; }
    public required MemoryLocation Location { get; init; }
}

/// <summary>
/// Memory location for a fusion variable
/// </summary>
public enum MemoryLocation
{
    Input,      // Input tensor
    Output,     // Output tensor
    Temporary,  // Temporary buffer in shared memory
    Register    // Stored in registers
}

/// <summary>
/// Memory layout for fusion operations
/// </summary>
public record MemoryLayout
{
    public required TensorLayout TensorLayout { get; init; }
    public required int SharedMemoryBytes { get; init; }
    public required int RegisterBytes { get; init; }
}

/// <summary>
/// Compute requirements for fusion operations
/// </summary>
public record ComputeRequirements
{
    public required int ThreadBlocks { get; init; }
    public required int ThreadsPerBlock { get; init; }
    public required bool RequiresSharedMemory { get; init; }
    public required bool RequiresAtomicOps { get; init; }
}

/// <summary>
/// Dataflow graph for fusion IR
/// </summary>
public class FusionDataflowGraph
{
    private readonly HashSet<FusionVariable> _variables = new();
    private readonly HashSet<FusionOpNode> _operations = new();
    private readonly List<(object From, object To)> _edges = new();

    public IReadOnlySet<FusionVariable> Variables => _variables;
    public IReadOnlySet<FusionOpNode> Operations => _operations;

    public void AddVariable(FusionVariable variable)
    {
        _variables.Add(variable);
    }

    public void AddOperation(FusionOpNode operation)
    {
        _operations.Add(operation);
    }

    public void AddEdge(object from, object to)
    {
        _edges.Add((from, to));
    }

    public bool HasCycles()
    {
        // Build adjacency list
        var adjacency = new Dictionary<object, HashSet<object>>();
        var allNodes = _variables.Concat<object>(_operations);

        foreach (var node in allNodes)
        {
            adjacency[node] = new HashSet<object>();
        }

        foreach (var (from, to) in _edges)
        {
            adjacency[from].Add(to);
        }

        // Detect cycles using DFS
        var visited = new HashSet<object>();
        var recursionStack = new HashSet<object>();

        foreach (var node in allNodes)
        {
            if (HasCycleDFS(node, visited, recursionStack, adjacency))
                return true;
        }

        return false;
    }

    private bool HasCycleDFS(
        object node,
        HashSet<object> visited,
        HashSet<object> recursionStack,
        Dictionary<object, HashSet<object>> adjacency)
    {
        if (recursionStack.Contains(node))
            return true;

        if (visited.Contains(node))
            return false;

        visited.Add(node);
        recursionStack.Add(node);

        if (adjacency.TryGetValue(node, out var neighbors))
        {
            foreach (var neighbor in neighbors)
            {
                if (HasCycleDFS(neighbor, visited, recursionStack, adjacency))
                    return true;
            }
        }

        recursionStack.Remove(node);
        return false;
    }
}
