# Spec: ElasticWorker Core Implementation

## Overview
Implement the ElasticWorker class which represents a client-side worker that connects to the ElasticCoordinator, handles rescaling events, and manages the training state synchronization lifecycle.

## Deliverables

**File:** `src/MachineLearning/Distributed/Worker/ElasticWorker.cs`
```csharp
namespace MachineLearning.Distributed.Worker;

using MachineLearning.Distributed.Configuration;
using MachineLearning.Distributed.Models;

/// <summary>
/// Worker that adapts to topology changes in the training cluster
/// </summary>
public class ElasticWorker : IDisposable
{
    private readonly ElasticTrainingConfig _config;
    private readonly WorkerId _workerId;
    private WorkerMetadata _metadata;
    private ClusterTopology? _currentTopology;
    private GlobalTrainingState? _currentState;
    private Timer? _heartbeatTimer;
    private bool _isRescaling;
    private bool _isConnected;
    private bool _isDisposed;

    /// <summary>
    /// Event fired when a rescaling event is received
    /// </summary>
    public event Action<RescalingEvent>? RescalingRequested;

    /// <summary>
    /// Event fired when the worker successfully synchronizes state
    /// </summary>
    public event Action<GlobalTrainingState>? StateSynchronized;

    /// <summary>
    /// Gets the worker's unique identifier
    /// </summary>
    public WorkerId WorkerId => _workerId;

    /// <summary>
    /// Gets the current cluster topology
    /// </summary>
    public ClusterTopology? CurrentTopology => _currentTopology;

    /// <summary>
    /// Gets the current training state
    /// </summary>
    public GlobalTrainingState? CurrentState => _currentState;

    /// <summary>
    /// Indicates if the worker is currently rescaling
    /// </summary>
    public bool IsRescaling => _isRescaling;

    public ElasticWorker(WorkerId workerId, ElasticTrainingConfig config)
    {
        _workerId = workerId;
        _config = config;
        _config.Validate();

        _metadata = new WorkerMetadata
        {
            WorkerId = workerId,
            Status = WorkerStatus.Joining,
            JoinTime = DateTime.UtcNow,
            LastHeartbeat = DateTime.UtcNow,
            Rank = -1,
            LocalWorldSize = 1
        };
    }

    /// <summary>
    /// Connect to coordinator and register the worker
    /// </summary>
    public void ConnectToCluster(string coordinatorAddress)
    {
        if (_isConnected)
        {
            throw new InvalidOperationException("Worker is already connected to the cluster");
        }

        // In a full implementation, this would establish RPC communication
        // For now, we'll simulate the connection

        _metadata.Status = WorkerStatus.Joining;
        _isConnected = true;

        StartHeartbeat();
    }

    /// <summary>
    /// Disconnect from the coordinator (graceful shutdown)
    /// </summary>
    public async Task DisconnectAsync()
    {
        if (!_isConnected)
        {
            return;
        }

        _metadata.Status = WorkerStatus.Leaving;

        // In a full implementation, this would send unregister request to coordinator
        // For now, we'll simulate the disconnection

        StopHeartbeat();
        _isConnected = false;

        await Task.CompletedTask;
    }

    /// <summary>
    /// Handle a rescaling event received from the coordinator
    /// </summary>
    public async Task OnRescalingEventAsync(RescalingEvent evt)
    {
        _isRescaling = true;

        try
        {
            // Notify subscribers of the rescaling request
            RescalingRequested?.Invoke(evt);

            if (_config.UseSynchronousRescaling)
            {
                // Wait for external code to handle rescaling before continuing
                await WaitForRescalingCompletionAsync();
            }
            else
            {
                // For asynchronous mode, just update state and continue
                await UpdateTopologyAsync(evt.NewTopology);
            }
        }
        finally
        {
            _isRescaling = false;
        }
    }

    /// <summary>
    /// Update the local topology based on new cluster state
    /// </summary>
    public async Task UpdateTopologyAsync(ClusterTopology newTopology)
    {
        _currentTopology = newTopology;

        // Update local rank if worker is in the new topology
        var rank = newTopology.Workers.IndexOf(_workerId);
        if (rank >= 0)
        {
            _metadata.Rank = rank;
        }

        await Task.CompletedTask;
    }

    /// <summary>
    /// Synchronize the local training state with the global state
    /// </summary>
    public async Task SynchronizeStateAsync(GlobalTrainingState state)
    {
        _currentState = state;

        // In a full implementation, this would:
        // 1. Apply optimizer state from coordinator
        // 2. Update local learning rate
        // 3. Update epoch/step counters

        StateSynchronized?.Invoke(state);

        await Task.CompletedTask;
    }

    /// <summary>
    /// Resume training after rescaling completes
    /// </summary>
    public async Task ResumeTrainingAsync()
    {
        if (!_isRescaling)
        {
            return;
        }

        _metadata.Status = WorkerStatus.Active;

        await Task.CompletedTask;
    }

    /// <summary>
    /// Redistribute data according to the provided plan
    /// </summary>
    public async Task RedistributeDataAsync(DataRedistributionPlan plan)
    {
        // In a full implementation, this would:
        // 1. Receive data shards from source workers
        // 2. Send data shards to destination workers
        // 3. Validate redistribution completeness

        await Task.CompletedTask;
    }

    /// <summary>
    /// Update the worker's local training state
    /// </summary>
    public void UpdateTrainingState(int epoch, int step, float learningRate)
    {
        if (_currentState != null)
        {
            _currentState.CurrentEpoch = epoch;
            _currentState.CurrentStep = step;
            _currentState.LearningRate = learningRate;
        }
    }

    /// <summary>
    /// Check if this worker should handle a specific data shard
    /// </summary>
    public bool ShouldHandleShard(int shardIndex, int totalShards)
    {
        if (_currentTopology == null || _metadata.Rank < 0)
        {
            return false;
        }

        // Simple hash-based sharding
        return shardIndex % _currentTopology.WorldSize == _metadata.Rank;
    }

    /// <summary>
    /// Get the current worker metadata
    /// </summary>
    public WorkerMetadata GetMetadata()
    {
        return new WorkerMetadata
        {
            WorkerId = _metadata.WorkerId,
            Status = _metadata.Status,
            JoinTime = _metadata.JoinTime,
            LastHeartbeat = _metadata.LastHeartbeat,
            Rank = _metadata.Rank,
            LocalWorldSize = _metadata.LocalWorldSize,
            CustomAttributes = new Dictionary<string, string>(_metadata.CustomAttributes)
        };
    }

    private void StartHeartbeat()
    {
        var interval = _config.WorkerHeartbeatTimeoutMs / 3;
        _heartbeatTimer = new Timer(
            _ => SendHeartbeat(),
            null,
            interval,
            interval
        );
    }

    private void StopHeartbeat()
    {
        _heartbeatTimer?.Dispose();
        _heartbeatTimer = null;
    }

    private void SendHeartbeat()
    {
        if (!_isConnected || _isDisposed)
        {
            return;
        }

        _metadata.LastHeartbeat = DateTime.UtcNow;

        // In a full implementation, this would send heartbeat to coordinator
        // For now, we'll just update the local timestamp
    }

    private async Task WaitForRescalingCompletionAsync()
    {
        // In a full implementation, this would use a synchronization primitive
        // to wait for external code to complete the rescaling operation

        // Placeholder: wait for the configured timeout
        await Task.Delay(100);
    }

    public void Dispose()
    {
        if (_isDisposed)
        {
            return;
        }

        _isDisposed = true;
        StopHeartbeat();
        GC.SuppressFinalize(this);
    }
}
```

**File:** `src/MachineLearning/Distributed/Models/DataRedistributionPlan.cs`
```csharp
namespace MachineLearning.Distributed.Models;

/// <summary>
/// Plan for redistributing data during a topology change
/// </summary>
public class DataRedistributionPlan
{
    /// <summary>
    /// List of transfers to execute
    /// </summary>
    public List<DataTransfer> Transfers { get; set; } = new();

    /// <summary>
    /// Worker-specific redistribution assignments
    /// </summary>
    public Dictionary<WorkerId, List<DataShard>> WorkerAssignments { get; set; } = new();

    /// <summary>
    /// Total number of shards in the redistribution
    /// </summary>
    public int TotalShards { get; set; }
}

/// <summary>
/// Represents a single data transfer between workers
/// </summary>
public record DataTransfer
{
    public WorkerId SourceWorker { get; init; } = null!;
    public WorkerId DestinationWorker { get; init; } = null!;
    public DataShard Shard { get; init; } = null!;
    public int Priority { get; init; }
    public DateTime EstimatedCompletionTime { get; init; }
}
```

## Implementation Notes

1. State Management: Worker maintains its local state (topology, training state) that gets updated during rescaling
2. Heartbeat: Periodic timer sends heartbeats to coordinator to maintain liveness
3. Event-Driven: Uses C# events to notify external code of rescaling requests and state synchronization
4. Graceful Shutdown: Supports clean disconnection and cleanup
5. Synchronous/Asynchronous: Supports both modes based on configuration

## Dependencies
- Configuration and data models from spec_elastic_config_models.md
- ElasticCoordinator from spec_elastic_coordinator_core.md (for RPC communication in future)

## Estimated Effort
~45 minutes

## Success Criteria
- Worker can connect to coordinator and register successfully
- Heartbeat mechanism updates timestamp correctly
- Rescaling events trigger appropriate notifications
- State synchronization updates local state correctly
- Graceful disconnection works properly
- All public events are fired with appropriate data
- Thread safety is maintained during rescaling operations
