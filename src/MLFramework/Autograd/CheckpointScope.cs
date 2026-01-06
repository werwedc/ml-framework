using System.Collections.Concurrent;

namespace MLFramework.Autograd;

/// <summary>
/// Represents a scope in which tensor operations can be checkpointed.
/// Checkpoint scopes allow selective activation storage to trade computation for memory.
/// </summary>
public class CheckpointScope : IDisposable
{
    private static readonly ConcurrentDictionary<string, CheckpointScope> _activeScopes = new();
    private readonly object _lock = new object();
    private bool _disposed = false;

    /// <summary>
    /// Gets the name of this checkpoint scope.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets or sets whether checkpointing is enabled for this scope.
    /// </summary>
    public bool IsEnabled { get; set; }

    /// <summary>
    /// Gets or sets whether recomputation is used for missing activations.
    /// </summary>
    public bool UseRecomputation { get; set; }

    /// <summary>
    /// Gets the list of checkpointed nodes in this scope.
    /// </summary>
    public List<CheckpointNode> CheckpointedNodes { get; }

    /// <summary>
    /// Initializes a new instance of the CheckpointScope class.
    /// </summary>
    /// <param name="name">The name of the checkpoint scope.</param>
    /// <param name="enabled">Whether checkpointing is enabled by default.</param>
    public CheckpointScope(string name, bool enabled = true)
    {
        if (string.IsNullOrEmpty(name))
            throw new ArgumentException("Name cannot be null or empty", nameof(name));

        Name = name;
        IsEnabled = enabled;
        UseRecomputation = true;
        CheckpointedNodes = new List<CheckpointNode>();

        _activeScopes.TryAdd(Name, this);
    }

    /// <summary>
    /// Gets the active checkpoint scope with the specified name.
    /// </summary>
    /// <param name="name">The name of the checkpoint scope.</param>
    /// <returns>The checkpoint scope if found, null otherwise.</returns>
    public static CheckpointScope? GetActiveScope(string name)
    {
        _activeScopes.TryGetValue(name, out var scope);
        return scope;
    }

    /// <summary>
    /// Gets all active checkpoint scopes.
    /// </summary>
    /// <returns>A list of all active checkpoint scopes.</returns>
    public static List<CheckpointScope> GetAllActiveScopes()
    {
        return _activeScopes.Values.ToList();
    }

    /// <summary>
    /// Registers a checkpoint node with this scope.
    /// </summary>
    /// <param name="node">The checkpoint node to register.</param>
    public void RegisterNode(CheckpointNode node)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CheckpointScope));

        if (node == null)
            throw new ArgumentNullException(nameof(node));

        lock (_lock)
        {
            CheckpointedNodes.Add(node);
        }
    }

    /// <summary>
    /// Clears all checkpointed activations in this scope.
    /// </summary>
    public void ClearCheckpoints()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CheckpointScope));

        lock (_lock)
        {
            foreach (var node in CheckpointedNodes)
            {
                node.ClearSavedActivations();
            }
        }
    }

    /// <summary>
    /// Gets the total memory usage of saved activations in this scope (in MB).
    /// </summary>
    /// <returns>The memory usage in megabytes.</returns>
    public double GetMemoryUsageMB()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CheckpointScope));

        lock (_lock)
        {
            return CheckpointedNodes.Sum(node => node.GetSavedMemorySize()) / (1024.0 * 1024.0);
        }
    }

    /// <summary>
    /// Disposes of the checkpoint scope and its resources.
    /// </summary>
    public void Dispose()
    {
        lock (_lock)
        {
            if (!_disposed)
            {
                ClearCheckpoints();
                _activeScopes.TryRemove(Name, out _);
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }
    }

    /// <summary>
    /// Finalizer for CheckpointScope.
    /// </summary>
    ~CheckpointScope()
    {
        Dispose();
    }
}
