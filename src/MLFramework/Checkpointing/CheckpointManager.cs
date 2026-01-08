namespace MLFramework.Checkpointing;

/// <summary>
/// Core checkpoint manager for activation checkpointing
/// </summary>
public class CheckpointManager : IDisposable
{
    private class CheckpointEntry
    {
        public string LayerId { get; set; } = string.Empty;
        public Tensor Activation { get; set; } = null!;
        public long MemorySize { get; set; }
        public DateTime CreatedAt { get; set; }
        public int AccessCount { get; set; }
    }

    private readonly Dictionary<string, CheckpointEntry> _checkpoints;
    private readonly object _lock = new object();
    private long _totalMemoryUsed;
    private long _peakMemoryUsed;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of CheckpointManager
    /// </summary>
    public CheckpointManager()
    {
        _checkpoints = new Dictionary<string, CheckpointEntry>();
        _totalMemoryUsed = 0;
        _peakMemoryUsed = 0;
        _disposed = false;
    }

    /// <summary>
    /// Registers a checkpoint for the given layer ID
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="activation">The activation tensor to checkpoint</param>
    /// <exception cref="ArgumentException">Thrown if layerId already exists</exception>
    public void RegisterCheckpoint(string layerId, Tensor activation)
    {
        ThrowIfDisposed();

        if (string.IsNullOrEmpty(layerId))
        {
            throw new ArgumentException("Layer ID cannot be null or empty", nameof(layerId));
        }

        if (activation == null)
        {
            throw new ArgumentNullException(nameof(activation));
        }

        lock (_lock)
        {
            if (_checkpoints.ContainsKey(layerId))
            {
                throw new ArgumentException($"Layer ID '{layerId}' is already registered", nameof(layerId));
            }

            var memorySize = activation.SizeInBytes;
            _totalMemoryUsed += memorySize;

            if (_totalMemoryUsed > _peakMemoryUsed)
            {
                _peakMemoryUsed = _totalMemoryUsed;
            }

            _checkpoints[layerId] = new CheckpointEntry
            {
                LayerId = layerId,
                Activation = activation,
                MemorySize = memorySize,
                CreatedAt = DateTime.UtcNow,
                AccessCount = 0
            };
        }
    }

    /// <summary>
    /// Retrieves a checkpointed activation or recomputes it if not stored
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="recomputeFunc">Function to recompute the activation if needed</param>
    /// <returns>The activation tensor</returns>
    /// <exception cref="KeyNotFoundException">Thrown if layerId not found and no recomputeFunc provided</exception>
    public Tensor RetrieveOrRecompute(string layerId, Func<Tensor>? recomputeFunc = null)
    {
        ThrowIfDisposed();

        if (string.IsNullOrEmpty(layerId))
        {
            throw new ArgumentException("Layer ID cannot be null or empty", nameof(layerId));
        }

        lock (_lock)
        {
            if (_checkpoints.TryGetValue(layerId, out var entry))
            {
                entry.AccessCount++;
                return entry.Activation;
            }

            if (recomputeFunc != null)
            {
                var recomputed = recomputeFunc();
                if (recomputed == null)
                {
                    throw new InvalidOperationException("Recompute function returned null");
                }

                // Register the recomputed checkpoint
                var memorySize = recomputed.SizeInBytes;
                _totalMemoryUsed += memorySize;

                if (_totalMemoryUsed > _peakMemoryUsed)
                {
                    _peakMemoryUsed = _totalMemoryUsed;
                }

                _checkpoints[layerId] = new CheckpointEntry
                {
                    LayerId = layerId,
                    Activation = recomputed,
                    MemorySize = memorySize,
                    CreatedAt = DateTime.UtcNow,
                    AccessCount = 1
                };

                return recomputed;
            }

            throw new KeyNotFoundException($"Checkpoint not found for layer: {layerId}");
        }
    }

    /// <summary>
    /// Checks if a checkpoint exists for the given layer ID
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <returns>True if checkpoint exists, false otherwise</returns>
    public bool HasCheckpoint(string layerId)
    {
        ThrowIfDisposed();

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
        ThrowIfDisposed();

        lock (_lock)
        {
            foreach (var entry in _checkpoints.Values)
            {
                entry.Activation?.Dispose();
            }

            _checkpoints.Clear();
            _totalMemoryUsed = 0;
        }
    }

    /// <summary>
    /// Gets memory statistics for current checkpoints
    /// </summary>
    /// <returns>Memory statistics including size, count, and peak usage</returns>
    public MemoryStats GetMemoryStats()
    {
        ThrowIfDisposed();

        lock (_lock)
        {
            return new MemoryStats
            {
                CurrentMemoryUsed = _totalMemoryUsed,
                PeakMemoryUsed = _peakMemoryUsed,
                CheckpointCount = _checkpoints.Count,
                AverageMemoryPerCheckpoint = _checkpoints.Count > 0 ? _totalMemoryUsed / _checkpoints.Count : 0,
                MemorySavings = 0 // This would be calculated based on total activations vs checkpointed activations
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
            ThrowIfDisposed();
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

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(CheckpointManager));
        }
    }
}
