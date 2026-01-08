using System;

namespace MLFramework.Checkpointing.Extensions;

/// <summary>
/// Wrapper for checkpointed modules
/// </summary>
/// <typeparam name="T">Type of the underlying module</typeparam>
public class CheckpointedModuleWrapper<T> : ICheckpointedModule<T>
    where T : class
{
    private readonly T _module;
    private readonly string _layerId;
    private readonly CheckpointConfig _config;
    private readonly CheckpointManager _checkpointManager;
    private readonly RecomputationEngine _recomputeEngine;
    private bool _checkpointingEnabled;

    /// <summary>
    /// Initializes a new instance of CheckpointedModuleWrapper
    /// </summary>
    /// <param name="module">The module to wrap</param>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="config">Checkpoint configuration (optional, uses default if null)</param>
    public CheckpointedModuleWrapper(
        T module,
        string layerId,
        CheckpointConfig? config = null)
    {
        _module = module ?? throw new ArgumentNullException(nameof(module));
        _layerId = layerId ?? throw new ArgumentNullException(nameof(layerId));
        _config = config ?? CheckpointConfig.Default;
        _checkpointManager = new CheckpointManager();
        _recomputeEngine = new RecomputationEngine();
        _checkpointingEnabled = true;
    }

    /// <summary>
    /// Gets the underlying module
    /// </summary>
    public T Module => _module;

    /// <summary>
    /// Gets the layer ID
    /// </summary>
    public string LayerId => _layerId;

    /// <summary>
    /// Gets the checkpoint configuration
    /// </summary>
    public CheckpointConfig Config => _config;

    /// <summary>
    /// Enables checkpointing
    /// </summary>
    public void EnableCheckpointing()
    {
        _checkpointingEnabled = true;
    }

    /// <summary>
    /// Disables checkpointing
    /// </summary>
    public void DisableCheckpointing()
    {
        _checkpointingEnabled = false;
    }

    /// <summary>
    /// Gets checkpointing statistics
    /// </summary>
    public CheckpointStatistics GetStatistics()
    {
        var memoryStats = _checkpointManager.GetMemoryStats();
        var recomputeStats = _recomputeEngine.GetStats();

        return new CheckpointStatistics
        {
            LayerId = _layerId,
            MemoryUsed = memoryStats.CurrentMemoryUsed,
            PeakMemoryUsed = memoryStats.PeakMemoryUsed,
            RecomputationCount = recomputeStats.TotalRecomputations,
            RecomputationTimeMs = recomputeStats.TotalRecomputationTimeMs,
            IsCheckpointingEnabled = _checkpointingEnabled,
            CheckpointCount = memoryStats.CheckpointCount,
            Timestamp = DateTime.UtcNow
        };
    }

    /// <summary>
    /// Disposes the wrapper and releases resources
    /// </summary>
    public void Dispose()
    {
        _checkpointManager.Dispose();
        _recomputeEngine.Dispose();
    }
}
