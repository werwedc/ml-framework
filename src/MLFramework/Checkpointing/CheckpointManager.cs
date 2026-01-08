namespace MLFramework.Checkpointing;

/// <summary>
/// Core checkpoint manager (stub implementation)
/// </summary>
public class CheckpointManager : IDisposable
{
    private readonly Dictionary<string, Tensor> _checkpoints;
    private readonly object _lock = new object();
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of CheckpointManager
    /// </summary>
    public CheckpointManager()
    {
        _checkpoints = new Dictionary<string, Tensor>();
        _disposed = false;
    }

    /// <summary>
    /// Registers a checkpoint for the given layer ID
    /// </summary>
    public void RegisterCheckpoint(string layerId, Tensor activation)
    {
        lock (_lock)
        {
            _checkpoints[layerId] = activation;
        }
    }

    /// <summary>
    /// Retrieves a checkpointed activation or recomputes it if not stored
    /// </summary>
    public Tensor RetrieveOrRecompute(string layerId, Func<Tensor>? recomputeFunc = null)
    {
        lock (_lock)
        {
            if (_checkpoints.TryGetValue(layerId, out var activation))
            {
                return activation;
            }

            if (recomputeFunc != null)
            {
                var recomputed = recomputeFunc();
                _checkpoints[layerId] = recomputed;
                return recomputed;
            }

            throw new KeyNotFoundException($"Checkpoint not found for layer: {layerId}");
        }
    }

    /// <summary>
    /// Checks if a checkpoint exists for the given layer ID
    /// </summary>
    public bool HasCheckpoint(string layerId)
    {
        lock (_lock)
        {
            return _checkpoints.ContainsKey(layerId);
        }
    }

    /// <summary>
    /// Clears all stored checkpoints and releases memory
    /// </summary>
    public void ClearCheckpoints()
    {
        lock (_lock)
        {
            _checkpoints.Clear();
        }
    }

    /// <summary>
    /// Gets memory statistics for current checkpoints
    /// </summary>
    public MemoryStats GetMemoryStats()
    {
        lock (_lock)
        {
            return new MemoryStats
            {
                TotalMemoryUsed = _checkpoints.Values.Sum(t => t.SizeInBytes),
                PeakMemoryUsed = _checkpoints.Values.Sum(t => t.SizeInBytes),
                CheckpointCount = _checkpoints.Count
            };
        }
    }

    /// <summary>
    /// Gets the number of currently stored checkpoints
    /// </summary>
    public int CheckpointCount
    {
        get
        {
            lock (_lock)
            {
                return _checkpoints.Count;
            }
        }
    }

    /// <summary>
    /// Disposes the manager and releases all resources
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            ClearCheckpoints();
            _disposed = true;
        }
    }
}
