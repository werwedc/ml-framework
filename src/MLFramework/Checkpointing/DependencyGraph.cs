namespace MLFramework.Checkpointing;

/// <summary>
/// Tracks dependencies between layers for optimal recomputation ordering
/// </summary>
public class DependencyGraph
{
    private readonly Dictionary<string, HashSet<string>> _dependencies;
    private readonly object _lock = new object();

    /// <summary>
    /// Initializes a new instance of DependencyGraph
    /// </summary>
    public DependencyGraph()
    {
        _dependencies = new Dictionary<string, HashSet<string>>();
    }

    /// <summary>
    /// Adds a dependency relationship
    /// </summary>
    /// <param name="layerId">Layer that depends on other layers</param>
    /// <param name="dependsOn">Layers that this layer depends on</param>
    public void AddDependency(string layerId, IEnumerable<string> dependsOn)
    {
        if (string.IsNullOrWhiteSpace(layerId))
            throw new ArgumentException("Layer ID cannot be null or whitespace");
        if (dependsOn == null)
            throw new ArgumentNullException(nameof(dependsOn));

        lock (_lock)
        {
            if (!_dependencies.ContainsKey(layerId))
            {
                _dependencies[layerId] = new HashSet<string>();
            }

            foreach (var dep in dependsOn)
            {
                _dependencies[layerId].Add(dep);
            }
        }
    }

    /// <summary>
    /// Gets layers in topological order (dependencies first)
    /// </summary>
    /// <param name="layerIds">Layers to order</param>
    /// <returns>Layers in topological order</returns>
    public List<string> GetTopologicalOrder(IEnumerable<string> layerIds)
    {
        if (layerIds == null)
            throw new ArgumentNullException(nameof(layerIds));

        var layers = layerIds.ToList();
        if (layers.Count == 0)
            return new List<string>();

        lock (_lock)
        {
            // Build a subgraph with only the requested layers and their dependencies
            var subgraph = new Dictionary<string, HashSet<string>>();
            var visited = new HashSet<string>();

            foreach (var layerId in layers)
            {
                BuildSubgraph(layerId, subgraph, visited);
            }

            // Perform topological sort using Kahn's algorithm
            var inDegree = new Dictionary<string, int>();
            var queue = new Queue<string>();
            var result = new List<string>();

            // Calculate in-degree for each node
            foreach (var node in subgraph.Keys)
            {
                inDegree[node] = 0;
            }

            foreach (var node in subgraph.Keys)
            {
                foreach (var dep in subgraph[node])
                {
                    if (inDegree.ContainsKey(dep))
                    {
                        inDegree[node]++;
                    }
                }
            }

            // Enqueue nodes with in-degree 0
            foreach (var node in inDegree.Where(kvp => kvp.Value == 0))
            {
                queue.Enqueue(node.Key);
            }

            // Process nodes
            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                result.Add(current);

                foreach (var neighbor in subgraph.Keys.Where(n => subgraph[n].Contains(current)))
                {
                    inDegree[neighbor]--;
                    if (inDegree[neighbor] == 0)
                    {
                        queue.Enqueue(neighbor);
                    }
                }
            }

            // If not all nodes are processed, there's a cycle
            if (result.Count != subgraph.Count)
            {
                throw new InvalidOperationException("Dependency graph contains a cycle");
            }

            // Filter to only requested layers (but preserve topological order)
            var resultHash = new HashSet<string>(result);
            return layers.Where(l => resultHash.Contains(l)).ToList();
        }
    }

    private void BuildSubgraph(string layerId, Dictionary<string, HashSet<string>> subgraph, HashSet<string> visited)
    {
        if (visited.Contains(layerId))
            return;

        visited.Add(layerId);

        if (!_dependencies.ContainsKey(layerId))
            return;

        if (!subgraph.ContainsKey(layerId))
        {
            subgraph[layerId] = new HashSet<string>();
        }

        foreach (var dep in _dependencies[layerId])
        {
            subgraph[layerId].Add(dep);
            BuildSubgraph(dep, subgraph, visited);
        }
    }

    /// <summary>
    /// Detects cycles in the dependency graph
    /// </summary>
    /// <returns>True if cycle detected, false otherwise</returns>
    public bool HasCycle()
    {
        lock (_lock)
        {
            var visited = new HashSet<string>();
            var recursionStack = new HashSet<string>();

            foreach (var node in _dependencies.Keys)
            {
                if (HasCycleUtil(node, visited, recursionStack))
                {
                    return true;
                }
            }

            return false;
        }
    }

    private bool HasCycleUtil(string node, HashSet<string> visited, HashSet<string> recursionStack)
    {
        if (recursionStack.Contains(node))
            return true;

        if (visited.Contains(node))
            return false;

        visited.Add(node);
        recursionStack.Add(node);

        if (_dependencies.ContainsKey(node))
        {
            foreach (var neighbor in _dependencies[node])
            {
                if (HasCycleUtil(neighbor, visited, recursionStack))
                {
                    return true;
                }
            }
        }

        recursionStack.Remove(node);
        return false;
    }

    /// <summary>
    /// Clears all dependencies
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            _dependencies.Clear();
        }
    }
}
