using RitterFramework.Core.Tensor;
using System.Collections.Concurrent;

namespace MLFramework.Autograd;

/// <summary>
/// Manages gradient checkpointing across the computational graph.
/// Provides automatic checkpoint selection and memory management.
/// </summary>
public class CheckpointManager
{
    private readonly ConcurrentDictionary<string, CheckpointScope> _registeredScopes;
    private readonly ConcurrentDictionary<int, CheckpointNode> _registeredNodes;
    private readonly object _lock = new object();
    private bool _disposed = false;

    /// <summary>
    /// Gets or sets the maximum memory usage in megabytes before triggering automatic checkpointing.
    /// </summary>
    public int MaxMemoryMB { get; set; }

    /// <summary>
    /// Gets or sets whether automatic checkpointing is enabled.
    /// </summary>
    public bool AutoCheckpoint { get; set; }

    /// <summary>
    /// Gets or sets the memory threshold (0.0 - 1.0) as a fraction of MaxMemoryMB.
    /// </summary>
    public float MemoryThreshold { get; set; }

    /// <summary>
    /// Gets the current total memory usage in megabytes.
    /// </summary>
    public double CurrentMemoryMB { get; private set; }

    /// <summary>
    /// Gets the count of registered checkpoint scopes.
    /// </summary>
    public int RegisteredScopeCount => _registeredScopes.Count;

    /// <summary>
    /// Gets the count of registered checkpoint nodes.
    /// </summary>
    public int RegisteredNodeCount => _registeredNodes.Count;

    /// <summary>
    /// Initializes a new instance of the CheckpointManager class.
    /// </summary>
    public CheckpointManager()
    {
        _registeredScopes = new ConcurrentDictionary<string, CheckpointScope>();
        _registeredNodes = new ConcurrentDictionary<int, CheckpointNode>();
        MaxMemoryMB = 1024; // 1 GB default
        AutoCheckpoint = false;
        MemoryThreshold = 0.8f;
        CurrentMemoryMB = 0;
    }

    /// <summary>
    /// Initializes a new instance of the CheckpointManager class with specific settings.
    /// </summary>
    /// <param name="maxMemoryMB">The maximum memory in megabytes.</param>
    /// <param name="autoCheckpoint">Whether automatic checkpointing is enabled.</param>
    /// <param name="memoryThreshold">The memory threshold as a fraction.</param>
    public CheckpointManager(int maxMemoryMB, bool autoCheckpoint = true, float memoryThreshold = 0.8f)
        : this()
    {
        MaxMemoryMB = maxMemoryMB;
        AutoCheckpoint = autoCheckpoint;
        MemoryThreshold = Math.Clamp(memoryThreshold, 0.0f, 1.0f);
    }

    /// <summary>
    /// Registers a checkpoint scope with the manager.
    /// </summary>
    /// <param name="scope">The checkpoint scope to register.</param>
    public void RegisterScope(CheckpointScope scope)
    {
        if (scope == null)
            throw new ArgumentNullException(nameof(scope));

        if (_disposed)
            throw new ObjectDisposedException(nameof(CheckpointManager));

        _registeredScopes.TryAdd(scope.Name, scope);
    }

    /// <summary>
    /// Unregisters a checkpoint scope from the manager.
    /// </summary>
    /// <param name="scope">The checkpoint scope to unregister.</param>
    public void UnregisterScope(CheckpointScope scope)
    {
        if (scope == null)
            throw new ArgumentNullException(nameof(scope));

        if (_disposed)
            throw new ObjectDisposedException(nameof(CheckpointManager));

        _registeredScopes.TryRemove(scope.Name, out _);
    }

    /// <summary>
    /// Registers a checkpoint node with the manager.
    /// </summary>
    /// <param name="node">The checkpoint node to register.</param>
    public void RegisterNode(CheckpointNode node)
    {
        if (node == null)
            throw new ArgumentNullException(nameof(node));

        if (_disposed)
            throw new ObjectDisposedException(nameof(CheckpointManager));

        _registeredNodes.TryAdd(node.GradFnId, node);

        // Update memory usage
        lock (_lock)
        {
            CurrentMemoryMB += node.GetSavedMemorySize() / (1024.0 * 1024.0);
        }

        // Trigger automatic checkpointing if enabled
        if (AutoCheckpoint && CurrentMemoryMB >= MaxMemoryMB * MemoryThreshold)
        {
            TriggerAutomaticCheckpointing();
        }
    }

    /// <summary>
    /// Determines whether a node should be checkpointed based on current memory usage.
    /// </summary>
    /// <param name="node">The node to evaluate.</param>
    /// <returns>True if the node should be checkpointed, false otherwise.</returns>
    public bool ShouldCheckpoint(GraphNode node)
    {
        if (node == null)
            throw new ArgumentNullException(nameof(node));

        if (_disposed)
            throw new ObjectDisposedException(nameof(CheckpointManager));

        lock (_lock)
        {
            if (!AutoCheckpoint)
                return false;

            return CurrentMemoryMB >= MaxMemoryMB * MemoryThreshold;
        }
    }

    /// <summary>
    /// Performs selective checkpointing on a list of nodes.
    /// Selects nodes that would save the most memory while maintaining reasonable compute overhead.
    /// </summary>
    /// <param name="nodes">The list of nodes to consider for checkpointing.</param>
    public void SelectiveCheckpointing(List<GraphNode> nodes)
    {
        if (nodes == null)
            throw new ArgumentNullException(nameof(nodes));

        if (_disposed)
            throw new ObjectDisposedException(nameof(CheckpointManager));

        // Estimate memory savings for each node
        var nodeEstimates = nodes
            .Where(n => n is CheckpointNode)
            .Cast<CheckpointNode>()
            .Select(n => new
            {
                Node = n,
                MemorySize = n.GetSavedMemorySize(),
                Children = n.Children.Count
            })
            .OrderByDescending(x => x.MemorySize)
            .ToList();

        // Select nodes that maximize memory savings
        // Try to distribute checkpoints evenly across the graph
        long targetMemorySavings = (long)((CurrentMemoryMB - MaxMemoryMB * (1.0 - MemoryThreshold)) * 1024 * 1024);
        long savedMemory = 0;
        int lastCheckpointIndex = -100;

        foreach (var estimate in nodeEstimates)
        {
            if (savedMemory >= targetMemorySavings)
                break;

            // Avoid checkpointing nodes too close together (to minimize recompute overhead)
            int currentIndex = nodes.IndexOf(estimate.Node);
            if (currentIndex - lastCheckpointIndex > 2)
            {
                estimate.Node.IsCheckpoint = true;
                savedMemory += estimate.MemorySize;
                lastCheckpointIndex = currentIndex;
            }
        }
    }

    /// <summary>
    /// Clears all checkpoints from all registered scopes.
    /// </summary>
    public void ClearAllCheckpoints()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CheckpointManager));

        foreach (var scope in _registeredScopes.Values)
        {
            scope.ClearCheckpoints();
        }

        lock (_lock)
        {
            CurrentMemoryMB = 0;
        }
    }

    /// <summary>
    /// Gets the memory usage of a specific scope in megabytes.
    /// </summary>
    /// <param name="scopeName">The name of the scope.</param>
    /// <returns>The memory usage in megabytes.</returns>
    public double GetScopeMemoryUsage(string scopeName)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CheckpointManager));

        if (_registeredScopes.TryGetValue(scopeName, out var scope))
        {
            return scope.GetMemoryUsageMB();
        }

        return 0;
    }

    /// <summary>
    /// Gets the total memory usage across all scopes in megabytes.
    /// </summary>
    /// <returns>The total memory usage in megabytes.</returns>
    public double GetTotalMemoryUsage()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CheckpointManager));

        return _registeredScopes.Values.Sum(s => s.GetMemoryUsageMB());
    }

    /// <summary>
    /// Triggers automatic checkpointing based on current memory state.
    /// </summary>
    private void TriggerAutomaticCheckpointing()
    {
        lock (_lock)
        {
            var allNodes = _registeredNodes.Values.ToList();
            SelectiveCheckpointing(allNodes.Select(n => (GraphNode)n).ToList());
        }
    }

    /// <summary>
    /// Gets all registered checkpoint scopes.
    /// </summary>
    /// <returns>A list of all registered checkpoint scopes.</returns>
    public List<CheckpointScope> GetAllScopes()
    {
        return _registeredScopes.Values.ToList();
    }

    /// <summary>
    /// Gets all registered checkpoint nodes.
    /// </summary>
    /// <returns>A list of all registered checkpoint nodes.</returns>
    public List<CheckpointNode> GetAllNodes()
    {
        return _registeredNodes.Values.ToList();
    }

    /// <summary>
    /// Disposes of the checkpoint manager and its resources.
    /// </summary>
    public void Dispose()
    {
        lock (_lock)
        {
            if (!_disposed)
            {
                ClearAllCheckpoints();
                _registeredScopes.Clear();
                _registeredNodes.Clear();
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }
    }

    /// <summary>
    /// Finalizer for CheckpointManager.
    /// </summary>
    ~CheckpointManager()
    {
        Dispose();
    }
}
