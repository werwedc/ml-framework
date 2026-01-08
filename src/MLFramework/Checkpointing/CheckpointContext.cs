using System;
using MLFramework.Checkpointing.Extensions;

namespace MLFramework.Checkpointing;

/// <summary>
/// Manages checkpointing context for a forward/backward pass
/// </summary>
public class CheckpointContext : IDisposable
{
    private readonly CheckpointManager _checkpointManager;
    private readonly RecomputationEngine _recomputeEngine;
    private readonly MemoryTracker _memoryTracker;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of CheckpointContext
    /// </summary>
    /// <param name="config">Checkpoint configuration</param>
    public CheckpointContext(CheckpointConfig config)
    {
        if (config == null)
            throw new ArgumentNullException(nameof(config));

        Config = config;
        _checkpointManager = new CheckpointManager();
        _recomputeEngine = new RecomputationEngine();
        _memoryTracker = new MemoryTracker();
        _disposed = false;
    }

    /// <summary>
    /// Gets the checkpoint manager
    /// </summary>
    public CheckpointManager CheckpointManager => _checkpointManager;

    /// <summary>
    /// Gets the recompute engine
    /// </summary>
    public RecomputationEngine RecomputeEngine => _recomputeEngine;

    /// <summary>
    /// Gets the memory tracker
    /// </summary>
    public MemoryTracker MemoryTracker => _memoryTracker;

    /// <summary>
    /// Gets the checkpoint configuration
    /// </summary>
    public CheckpointConfig Config { get; }

    /// <summary>
    /// Gets whether checkpointing is currently enabled
    /// </summary>
    public bool IsEnabled { get; private set; }

    /// <summary>
    /// Enters the checkpoint context (enables checkpointing)
    /// </summary>
    public void Enter()
    {
        ThrowIfDisposed();
        IsEnabled = true;
    }

    /// <summary>
    /// Exits the checkpoint context (disables checkpointing and cleans up)
    /// </summary>
    public void Exit()
    {
        ThrowIfDisposed();
        IsEnabled = false;
        _checkpointManager.ClearCheckpoints();
    }

    /// <summary>
    /// Gets checkpointing statistics
    /// </summary>
    /// <returns>Checkpointing statistics</returns>
    public CheckpointStatistics GetStatistics()
    {
        ThrowIfDisposed();

        var memoryStats = _checkpointManager.GetMemoryStats();
        var recomputeStats = _recomputeEngine.GetStats();

        return new CheckpointStatistics
        {
            MemoryUsed = memoryStats.CurrentMemoryUsed,
            PeakMemoryUsed = memoryStats.PeakMemoryUsed,
            RecomputationCount = recomputeStats.TotalRecomputations,
            RecomputationTimeMs = recomputeStats.TotalRecomputationTimeMs,
            IsCheckpointingEnabled = IsEnabled,
            CheckpointCount = memoryStats.CheckpointCount,
            Timestamp = DateTime.UtcNow
        };
    }

    /// <summary>
    /// Disposes the context and exits if still active
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            if (IsEnabled)
            {
                Exit();
            }
            _checkpointManager.Dispose();
            _recomputeEngine.Dispose();
            _disposed = true;
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(CheckpointContext));
    }
}
