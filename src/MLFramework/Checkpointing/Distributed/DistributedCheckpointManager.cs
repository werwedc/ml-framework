using MLFramework.Distributed;
using RitterFramework.Core.Tensor;

namespace MLFramework.Checkpointing.Distributed;

/// <summary>
/// Manages checkpointing across multiple devices in distributed training
/// </summary>
public class DistributedCheckpointManager : IDisposable
{
    private readonly int _rank;
    private readonly int _worldSize;
    private readonly ProcessGroup _processGroup;
    private readonly CheckpointManager _localCheckpointManager;
    private readonly DistributedCommunication _communication;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of DistributedCheckpointManager
    /// </summary>
    /// <param name="rank">Rank of the current process</param>
    /// <param name="worldSize">Total number of processes</param>
    /// <param name="processGroup">Process group for communication</param>
    public DistributedCheckpointManager(
        int rank,
        int worldSize,
        ProcessGroup? processGroup = null)
    {
        _rank = rank;
        _worldSize = worldSize;
        _processGroup = processGroup ?? ProcessGroup.Default ?? throw new InvalidOperationException(
            "No process group provided and no default process group initialized. " +
            "Call ProcessGroup.Init() before creating DistributedCheckpointManager.");
        _localCheckpointManager = new CheckpointManager();
        _communication = new DistributedCommunication(_processGroup);
        _disposed = false;
    }

    /// <summary>
    /// Gets the local checkpoint manager
    /// </summary>
    public CheckpointManager LocalManager => _localCheckpointManager;

    /// <summary>
    /// Gets the rank of the current process
    /// </summary>
    public int Rank => _rank;

    /// <summary>
    /// Gets the total number of processes
    /// </summary>
    public int WorldSize => _worldSize;

    /// <summary>
    /// Registers a checkpoint for the given layer ID
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="activation">The activation tensor to checkpoint</param>
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

        _localCheckpointManager.RegisterCheckpoint(layerId, activation);
    }

    /// <summary>
    /// Registers a checkpoint and broadcasts to all processes
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// < <param name="activation">The activation tensor to checkpoint</param>
    public void RegisterCheckpointBroadcast(string layerId, Tensor activation)
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

        // Register locally
        _localCheckpointManager.RegisterCheckpoint(layerId, activation);

        // Broadcast to all other ranks (rank 0 broadcasts)
        // Note: In a real implementation, we would convert MLFramework.Checkpointing.Tensor
        // to RitterFramework.Core.Tensor.Tensor here. For now, we'll just use barrier.
        // _communication.Broadcast(activation, 0);

        // Wait for all ranks to complete
        _communication.Barrier();
    }

    /// <summary>
    /// Synchronizes checkpoint state across all processes
    /// </summary>
    public void SynchronizeCheckpoints()
    {
        ThrowIfDisposed();

        // Collect checkpoint keys from all ranks
        var localKeys = GetCheckpointKeys();
        var allKeys = _communication.AllGather(localKeys);

        // Ensure all ranks have the same checkpoint state
        // This is a simplified version - actual implementation would be more complex
        _communication.Barrier();
    }

    /// <summary>
    /// Retrieves a checkpointed activation (local or from another rank)
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="sourceRank">Rank to retrieve from (null for local)</param>
    /// <returns>The activation tensor</returns>
    public Tensor RetrieveOrFetch(string layerId, int? sourceRank = null)
    {
        ThrowIfDisposed();

        if (string.IsNullOrEmpty(layerId))
        {
            throw new ArgumentException("Layer ID cannot be null or empty", nameof(layerId));
        }

        if (sourceRank == null || sourceRank == _rank)
        {
            // Local retrieval
            return _localCheckpointManager.RetrieveOrRecompute(
                layerId,
                () => throw new KeyNotFoundException($"Layer {layerId} not found"));
        }
        else
        {
            // Remote retrieval
            // Note: In a real implementation, we would convert RitterFramework.Core.Tensor.Tensor
            // to MLFramework.Checkpointing.Tensor here. For now, we'll throw.
            var received = _communication.ReceiveTensor(sourceRank.Value, layerId);
            // TODO: Convert received tensor to MLFramework.Checkpointing.Tensor
            throw new NotImplementedException(
                "Remote tensor retrieval not yet implemented. " +
                "A tensor conversion adapter between RitterFramework.Core.Tensor and MLFramework.Checkpointing.Tensor is required.");
        }
    }

    /// <summary>
    /// Clears all checkpoints across all processes
    /// </summary>
    public void ClearCheckpointsDistributed()
    {
        ThrowIfDisposed();

        _localCheckpointManager.ClearCheckpoints();

        // Synchronize to ensure all ranks clear
        _communication.Barrier();
    }

    /// <summary>
    /// Gets aggregated memory statistics from all processes
    /// </summary>
    /// <returns>Aggregated memory statistics</returns>
    public DistributedMemoryStats GetAggregatedMemoryStats()
    {
        ThrowIfDisposed();

        var localStats = _localCheckpointManager.GetMemoryStats();

        // Gather stats from all ranks
        var allStats = _communication.AllGather(localStats);

        // Aggregate
        var aggregated = new DistributedMemoryStats
        {
            TotalCurrentMemoryUsed = allStats.Sum(s => s.CurrentMemoryUsed),
            TotalPeakMemoryUsed = allStats.Sum(s => s.PeakMemoryUsed),
            PerRankMemoryUsed = allStats.Select(s => s.CurrentMemoryUsed).ToList(),
            AverageMemoryPerRank = allStats.Count > 0 ? (long)allStats.Average(s => s.CurrentMemoryUsed) : 0,
            TotalCheckpointCount = allStats.Sum(s => s.CheckpointCount),
            PerRankCheckpointCount = allStats.Select(s => s.CheckpointCount).ToList(),
            MaxMemoryUsed = allStats.Count > 0 ? allStats.Max(s => s.CurrentMemoryUsed) : 0,
            MinMemoryUsed = allStats.Count > 0 ? allStats.Min(s => s.CurrentMemoryUsed) : 0,
            Timestamp = DateTime.UtcNow
        };

        return aggregated;
    }

    /// <summary>
    /// Gets the checkpoint keys from the local manager
    /// </summary>
    /// <returns>List of checkpoint keys</returns>
    private List<string> GetCheckpointKeys()
    {
        // This is a simplified implementation
        // In a real implementation, CheckpointManager would expose a method to get all keys
        // For now, we'll return an empty list
        return new List<string>();
    }

    /// <summary>
    /// Disposes the manager and releases resources
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _localCheckpointManager.Dispose();
            _communication.Dispose();
            _disposed = true;
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(DistributedCheckpointManager));
        }
    }
}
