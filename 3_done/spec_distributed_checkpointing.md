# Spec: Distributed Training Support

## Overview
Implement support for checkpointing in distributed training scenarios, including synchronization across devices, communication-aware checkpointing for model parallelism, and cross-stage checkpointing for pipeline parallelism.

## Classes

### Location
`src/MLFramework/Checkpointing/Distributed/`

### Class: DistributedCheckpointManager

```csharp
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
        _processGroup = processGroup ?? ProcessGroup.World;
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
    public void RegisterCheckpoint(string layerId, Tensor activation);

    /// <summary>
    /// Registers a checkpoint and broadcasts to all processes
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="activation">The activation tensor to checkpoint</param>
    public void RegisterCheckpointBroadcast(string layerId, Tensor activation);

    /// <summary>
    /// Synchronizes checkpoint state across all processes
    /// </summary>
    public void SynchronizeCheckpoints();

    /// <summary>
    /// Retrieves a checkpointed activation (local or from another rank)
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="sourceRank">Rank to retrieve from (null for local)</param>
    /// <returns>The activation tensor</returns>
    public Tensor RetrieveOrFetch(string layerId, int? sourceRank = null);

    /// <summary>
    /// Clears all checkpoints across all processes
    /// </summary>
    public void ClearCheckpointsDistributed();

    /// <summary>
    /// Gets aggregated memory statistics from all processes
    /// </summary>
    /// <returns>Aggregated memory statistics</returns>
    public DistributedMemoryStats GetAggregatedMemoryStats();

    /// <summary>
    /// Disposes the manager and releases resources
    /// </summary>
    public void Dispose();
}
```

## Implementation Details

### RegisterCheckpointBroadcast

```csharp
public void RegisterCheckpointBroadcast(string layerId, Tensor activation)
{
    // Register locally
    _localCheckpointManager.RegisterCheckpoint(layerId, activation);

    // Broadcast to all other ranks
    _communication.Broadcast(activation, 0); // Rank 0 broadcasts

    // Wait for all ranks to complete
    _communication.Barrier();
}
```

### SynchronizeCheckpoints

```csharp
public void SynchronizeCheckpoints()
{
    // Collect checkpoint keys from all ranks
    var localKeys = _localCheckpointManager.GetCheckpointKeys();
    var allKeys = _communication.AllGather(localKeys);

    // Ensure all ranks have the same checkpoint state
    // This is a simplified version - actual implementation would be more complex
    _communication.Barrier();
}
```

### RetrieveOrFetch

```csharp
public Tensor RetrieveOrFetch(string layerId, int? sourceRank = null)
{
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
        return _communication.ReceiveTensor(sourceRank.Value, layerId);
    }
}
```

### GetAggregatedMemoryStats

```csharp
public DistributedMemoryStats GetAggregatedMemoryStats()
{
    var localStats = _localCheckpointManager.GetMemoryStats();

    // Gather stats from all ranks
    var allStats = _communication.AllGather(localStats);

    // Aggregate
    var aggregated = new DistributedMemoryStats
    {
        TotalCurrentMemoryUsed = allStats.Sum(s => s.CurrentMemoryUsed),
        TotalPeakMemoryUsed = allStats.Sum(s => s.PeakMemoryUsed),
        PerRankMemoryUsed = allStats.Select(s => s.CurrentMemoryUsed).ToList(),
        AverageMemoryPerRank = allStats.Average(s => s.CurrentMemoryUsed),
        TotalCheckpointCount = allStats.Sum(s => s.CheckpointCount)
    };

    return aggregated;
}
```

## Distributed Communication

### Class: DistributedCommunication

```csharp
namespace MLFramework.Checkpointing.Distributed;

/// <summary>
/// Handles communication between distributed processes
/// </summary>
public class DistributedCommunication : IDisposable
{
    private readonly ProcessGroup _processGroup;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of DistributedCommunication
    /// </summary>
    /// <param name="processGroup">Process group for communication</param>
    public DistributedCommunication(ProcessGroup processGroup)
    {
        _processGroup = processGroup ?? throw new ArgumentNullException(nameof(processGroup));
        _disposed = false;
    }

    /// <summary>
    /// Broadcasts a tensor to all processes
    /// </summary>
    /// <param name="tensor">Tensor to broadcast</param>
    /// <param name="sourceRank">Source rank</param>
    public void Broadcast(Tensor tensor, int sourceRank);

    /// <summary>
    /// Sends a tensor to a specific rank
    /// </summary>
    /// <param name="tensor">Tensor to send</param>
    /// <param name="destinationRank">Destination rank</param>
    /// <param name="tag">Tag for the message</param>
    public void Send(Tensor tensor, int destinationRank, int tag = 0);

    /// <summary>
    /// Receives a tensor from a specific rank
    /// </summary>
    /// <param name="sourceRank">Source rank</param>
    /// <param name="tag">Tag for the message</param>
    /// <returns>Received tensor</returns>
    public Tensor Receive(int sourceRank, int tag = 0);

    /// <summary>
    /// Receives a tensor with layer ID
    /// </summary>
    /// <param name="sourceRank">Source rank</param>
    /// <param name="layerId">Layer ID to receive</param>
    /// <returns>Received tensor</returns>
    public Tensor ReceiveTensor(int sourceRank, string layerId)
    {
        return Receive(sourceRank, HashCode.Combine(layerId));
    }

    /// <summary>
    /// All-gather operation
    /// </summary>
    /// <typeparam name="T">Type of data</typeparam>
    /// <param name="data">Local data</param>
    /// <returns>Data from all ranks</returns>
    public List<T> AllGather<T>(T data);

    /// <summary>
    /// Barrier operation - synchronizes all processes
    /// </summary>
    public void Barrier();

    /// <summary>
    /// Reduces data from all ranks (sum)
    /// </summary>
    /// <typeparam name="T">Type of data</typeparam>
    /// <param name="data">Local data</param>
    /// <returns>Reduced data</returns>
    public T Reduce<T>(T data);

    /// <summary>
    /// All-reduce operation
    /// </summary>
    /// <typeparam name="T">Type of data</typeparam>
    /// <param name="data">Local data</param>
    /// <returns>Reduced data on all ranks</returns>
    public T AllReduce<T>(T data);

    /// <summary>
    /// Disposes the communication and releases resources
    /// </summary>
    public void Dispose();
}
```

## Model Parallelism Checkpointing

### Class: ModelParallelCheckpointStrategy

```csharp
namespace MLFramework.Checkpointing.Distributed;

/// <summary>
/// Checkpointing strategy for model parallelism
/// </summary>
public class ModelParallelCheckpointStrategy
{
    private readonly DistributedCheckpointManager _checkpointManager;
    private readonly int _tensorParallelSize;
    private readonly Dictionary<string, int> _layerToRankMap;

    /// <summary>
    /// Initializes a new instance of ModelParallelCheckpointStrategy
    /// </summary>
    /// <param name="checkpointManager">Distributed checkpoint manager</param>
    /// <param name="tensorParallelSize">Tensor parallel size</param>
    /// <param name="layerToRankMap">Mapping of layers to ranks</param>
    public ModelParallelCheckpointStrategy(
        DistributedCheckpointManager checkpointManager,
        int tensorParallelSize,
        Dictionary<string, int>? layerToRankMap = null)
    {
        _checkpointManager = checkpointManager;
        _tensorParallelSize = tensorParallelSize;
        _layerToRankMap = layerToRankMap ?? new Dictionary<string, int>();
    }

    /// <summary>
    /// Determines whether to checkpoint a layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="activation">The activation tensor</param>
    /// <param name="isTensorParallel">Whether the layer is tensor parallel</param>
    /// <returns>True if should checkpoint, false otherwise</returns>
    public bool ShouldCheckpoint(
        string layerId,
        Tensor activation,
        bool isTensorParallel = false)
    {
        // If tensor parallel, checkpoint only on first rank
        if (isTensorParallel)
        {
            return _checkpointManager.Rank % _tensorParallelSize == 0;
        }

        // Otherwise, checkpoint based on strategy
        return true;
    }

    /// <summary>
    /// Gets the rank that should store the checkpoint for a layer
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <returns>Rank to store checkpoint</returns>
    public int GetCheckpointRank(string layerId)
    {
        return _layerToRankMap.TryGetValue(layerId, out var rank) ? rank : 0;
    }

    /// <summary>
    /// Registers a layer-rank mapping
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="rank">Rank to store checkpoint</param>
    public void RegisterLayerRank(string layerId, int rank)
    {
        _layerToRankMap[layerId] = rank;
    }
}
```

## Pipeline Parallelism Checkpointing

### Class: PipelineParallelCheckpointManager

```csummary>
/// Manages checkpointing for pipeline parallelism
/// </summary>
public class PipelineParallelCheckpointManager : IDisposable
{
    private readonly DistributedCheckpointManager _checkpointManager;
    private readonly int _numStages;
    private readonly int _currentStage;
    private readonly List<StageCheckpoint> _stageCheckpoints;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of PipelineParallelCheckpointManager
    /// </summary>
    /// <param name="checkpointManager">Distributed checkpoint manager</param>
    /// <param name="numStages">Number of pipeline stages</param>
    /// <param name="currentStage">Current pipeline stage</param>
    public PipelineParallelCheckpointManager(
        DistributedCheckpointManager checkpointManager,
        int numStages,
        int currentStage)
    {
        _checkpointManager = checkpointManager;
        _numStages = numStages;
        _currentStage = currentStage;
        _stageCheckpoints = new List<StageCheckpoint>();
        _disposed = false;
    }

    /// <summary>
    /// Registers a checkpoint for the current pipeline stage
    /// </summary>
    /// <param name="layerId">Unique identifier for the layer</param>
    /// <param name="activation">The activation tensor to checkpoint</param>
    /// <param name="isBoundary">Whether this is a stage boundary</param>
    public void RegisterStageCheckpoint(
        string layerId,
        Tensor activation,
        bool isBoundary = false)
    {
        var stageCheckpoint = new StageCheckpoint
        {
            LayerId = layerId,
            Activation = activation,
            Stage = _currentStage,
            IsBoundary = isBoundary,
            Timestamp = DateTime.UtcNow
        };

        _stageCheckpoints.Add(stageCheckpoint);

        // If it's a boundary, store it in distributed manager
        if (isBoundary)
        {
            _checkpointManager.RegisterCheckpoint(layerId, activation);
        }
    }

    /// <summary>
    /// Gets checkpoints for a specific stage
    /// </summary>
    /// <param name="stage">Stage to get checkpoints for</param>
    /// <returns>List of checkpoints</returns>
    public List<Tensor> GetStageCheckpoints(int stage)
    {
        return _stageCheckpoints
            .Where(sc => sc.Stage == stage)
            .Select(sc => sc.Activation)
            .ToList();
    }

    /// <summary>
    /// Gets boundary checkpoints for the next stage
    /// </summary>
    /// <returns>List of boundary checkpoints</returns>
    public List<Tensor> GetNextStageBoundaries()
    {
        return _stageCheckpoints
            .Where(sc => sc.IsBoundary && sc.Stage == _currentStage)
            .Select(sc => sc.Activation)
            .ToList();
    }

    /// <summary>
    /// Clears all stage checkpoints
    /// </summary>
    public void ClearStageCheckpoints()
    {
        _stageCheckpoints.Clear();
    }

    /// <summary>
    /// Disposes the manager and releases resources
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            ClearStageCheckpoints();
            _disposed = true;
        }
    }

    private class StageCheckpoint
    {
        public string LayerId { get; set; } = string.Empty;
        public Tensor Activation { get; set; } = null!;
        public int Stage { get; set; }
        public bool IsBoundary { get; set; }
        public DateTime Timestamp { get; set; }
    }
}
```

## Distributed Memory Statistics

### Class: DistributedMemoryStats

```csharp
namespace MLFramework.Checkpointing.Distributed;

/// <summary>
/// Memory statistics for distributed checkpointing
/// </summary>
public class DistributedMemoryStats
{
    /// <summary>
    /// Total current memory used across all ranks (in bytes)
    /// </summary>
    public long TotalCurrentMemoryUsed { get; set; }

    /// <summary>
    /// Total peak memory used across all ranks (in bytes)
    /// </summary>
    public long TotalPeakMemoryUsed { get; set; }

    /// <summary>
    /// Memory used per rank (in bytes)
    /// </summary>
    public List<long> PerRankMemoryUsed { get; set; } = new List<long>();

    /// <summary>
    /// Average memory per rank (in bytes)
    /// </summary>
    public long AverageMemoryPerRank { get; set; }

    /// <summary>
    /// Maximum memory used by any rank (in bytes)
    /// </summary>
    public long MaxMemoryUsed { get; set; }

    /// <summary>
    /// Minimum memory used by any rank (in bytes)
    /// </summary>
    public long MinMemoryUsed { get; set; }

    /// <summary>
    /// Total checkpoint count across all ranks
    /// </summary>
    public int TotalCheckpointCount { get; set; }

    /// <summary>
    /// Checkpoint count per rank
    /// </summary>
    public List<int> PerRankCheckpointCount { get; set; } = new List<int>();

    /// <summary>
    /// Timestamp when stats were collected
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Creates a string summary of the statistics
    /// </summary>
    /// <returns>Summary string</returns>
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine("Distributed Memory Statistics:");
        sb.AppendLine($"  Total Memory Used: {FormatBytes(TotalCurrentMemoryUsed)}");
        sb.AppendLine($"  Total Peak Memory: {FormatBytes(TotalPeakMemoryUsed)}");
        sb.AppendLine($"  Average Memory Per Rank: {FormatBytes(AverageMemoryPerRank)}");
        sb.AppendLine($"  Max Memory Used: {FormatBytes(MaxMemoryUsed)}");
        sb.AppendLine($"  Min Memory Used: {FormatBytes(MinMemoryUsed)}");
        sb.AppendLine($"  Total Checkpoints: {TotalCheckpointCount}");
        sb.AppendLine($"  Per-Rank Memory: [{string.Join(", ", PerRankMemoryUsed.Select(FormatBytes))}]");
        return sb.ToString();
    }

    private string FormatBytes(long bytes)
    {
        if (bytes < 1024) return $"{bytes}B";
        if (bytes < 1024 * 1024) return $"{bytes / 1024.0:F2}KB";
        if (bytes < 1024 * 1024 * 1024) return $"{bytes / (1024.0 * 1024):F2}MB";
        return $"{bytes / (1024.0 * 1024 * 1024):F2}GB";
    }
}
```

## Testing Requirements

### Unit Tests

1. **DistributedCheckpointManager Tests**
   - [ ] RegisterCheckpoint stores locally
   - [ ] RegisterCheckpointBroadcast broadcasts to all ranks
   - [ ] SynchronizeCheckpoints synchronizes state
   - [ ] RetrieveOrFetch retrieves local and remote checkpoints
   - [ ] ClearCheckpointsDistributed clears all checkpoints
   - [ ] GetAggregatedMemoryStats returns correct statistics

2. **DistributedCommunication Tests**
   - [ ] Broadcast sends data to all ranks
   - [ ] Send/Receive transfer data correctly
   - [ ] AllGather collects data from all ranks
   - [ ] Barrier synchronizes all ranks
   - [ ] Reduce performs reduction correctly
   - [ ] AllReduce performs all-reduce correctly

3. **ModelParallelCheckpointStrategy Tests**
   - [ ] ShouldCheckpoint handles tensor parallel layers correctly
   - [ ] ShouldCheckpoint handles non-tensor parallel layers correctly
   - [ ] GetCheckpointRank returns correct rank
   - [ ] RegisterLayerRank stores mapping correctly

4. **PipelineParallelCheckpointManager Tests**
   - [ ] RegisterStageCheckpoint stores checkpoints correctly
   - [ ] RegisterStageCheckpoint handles boundary checkpoints
   - [ ] GetStageCheckpoints returns correct checkpoints
   - [ ] GetNextStageBoundaries returns correct boundaries
   - [ ] ClearStageCheckpoints removes all checkpoints

5. **DistributedMemoryStats Tests**
   - [ ] ToString generates correct string
   - [ ] FormatBytes formats correctly for various sizes
   - [ ] All properties are set correctly

6. **Integration Tests**
   - [ ] End-to-end distributed checkpointing
   - [ ] Model parallelism with checkpointing
   - [ ] Pipeline parallelism with checkpointing
   - [ ] Synchronization across multiple ranks

7. **Edge Cases**
   - [ ] Handle single rank (world size 1)
   - [ ] Handle invalid rank values
   - [ ] Handle communication failures
   - [ ] Handle empty checkpoint lists

## Implementation Notes

1. **Communication Overhead**:
   - Minimize communication for checkpointing
   - Overlap communication with computation where possible
   - Use efficient communication primitives

2. **Synchronization**:
   - Ensure all ranks stay synchronized
   - Handle stragglers appropriately
   - Use barriers judiciously

3. **Error Handling**:
   - Handle communication failures gracefully
   - Provide recovery mechanisms
   - Maintain consistency across ranks

4. **Scalability**:
   - Scale to many ranks efficiently
   - Minimize memory overhead per rank
   - Avoid bottlenecks in communication

## Dependencies on Other Specs

This spec depends on:
- **Checkpoint Manager Core** (spec_1) for CheckpointManager
- **Memory Tracking System** (spec_3) for MemoryStats

## Estimated Implementation Time
45-60 minutes
