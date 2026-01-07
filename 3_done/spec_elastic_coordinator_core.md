# Spec: ElasticCoordinator Core Implementation

## Overview
Implement the ElasticCoordinator class which serves as the central coordination service for managing cluster membership, health monitoring, and triggering rescaling operations. The coordinator maintains the global view of the training cluster and communicates topology changes to workers.

## Deliverables

**File:** `src/MachineLearning/Distributed/Coordinator/ElasticCoordinator.cs`
```csharp
namespace MachineLearning.Distributed.Coordinator;

using MachineLearning.Distributed.Configuration;
using MachineLearning.Distributed.Enums;
using MachineLearning.Distributed.Models;

/// <summary>
/// Central coordinator that manages cluster membership and topology changes
/// </summary>
public class ElasticCoordinator : IDisposable
{
    private readonly ElasticTrainingConfig _config;
    private readonly ClusterTopology _currentTopology;
    private readonly Dictionary<WorkerId, WorkerMetadata> _workers;
    private readonly object _lock = new();
    private Timer? _healthCheckTimer;
    private Timer? _stabilityTimer;
    private List<WorkerId>? _pendingWorkers;
    private DateTime? _lastTopologyChange;
    private int _consecutiveFailures;

    /// <summary>
    /// Event fired when a rescaling operation is triggered
    /// </summary>
    public event Action<RescalingEvent>? RescalingTriggered;

    /// <summary>
    /// Event fired when a worker's status changes
    /// </summary>
    public event Action<WorkerId, WorkerStatus>? WorkerStatusChanged;

    public ElasticCoordinator(ElasticTrainingConfig config)
    {
        _config = config;
        _config.Validate();

        _currentTopology = new ClusterTopology();
        _workers = new Dictionary<WorkerId, WorkerMetadata>();
        _pendingWorkers = new List<WorkerId>();
    }

    /// <summary>
    /// Register a new worker to the training cluster
    /// </summary>
    public void RegisterWorker(WorkerMetadata metadata)
    {
        lock (_lock)
        {
            if (_workers.ContainsKey(metadata.WorkerId))
            {
                throw new InvalidOperationException($"Worker {metadata.WorkerId} is already registered");
            }

            metadata.Status = WorkerStatus.Joining;
            metadata.JoinTime = DateTime.UtcNow;
            metadata.LastHeartbeat = DateTime.UtcNow;

            _workers[metadata.WorkerId] = metadata;
            _pendingWorkers!.Add(metadata.WorkerId);

            WorkerStatusChanged?.Invoke(metadata.WorkerId, WorkerStatus.Joining);

            StartStabilityTimer();
        }
    }

    /// <summary>
    /// Unregister a worker from the cluster (graceful or failure)
    /// </summary>
    public void UnregisterWorker(WorkerId worker, bool graceful = true)
    {
        lock (_lock)
        {
            if (!_workers.TryGetValue(worker, out var metadata))
            {
                return;
            }

            var oldTopology = CloneTopology(_currentTopology);
            metadata.Status = graceful ? WorkerStatus.Leaving : WorkerStatus.Failed;
            WorkerStatusChanged?.Invoke(worker, metadata.Status);

            _currentTopology.RemoveWorker(worker);

            if (!graceful)
            {
                _consecutiveFailures++;
                CheckFailureTolerance();
            }

            TriggerRescalingIfStable(oldTopology);
        }
    }

    /// <summary>
    /// Update worker heartbeat timestamp
    /// </summary>
    public void UpdateHeartbeat(WorkerId worker)
    {
        lock (_lock)
        {
            if (_workers.TryGetValue(worker, out var metadata))
            {
                metadata.LastHeartbeat = DateTime.UtcNow;
            }
        }
    }

    /// <summary>
    /// Get current cluster topology
    /// </summary>
    public ClusterTopology GetClusterState()
    {
        lock (_lock)
        {
            return CloneTopology(_currentTopology);
        }
    }

    /// <summary>
    /// Get metadata for a specific worker
    /// </summary>
    public WorkerMetadata? GetWorkerMetadata(WorkerId worker)
    {
        lock (_lock)
        {
            return _workers.TryGetValue(worker, out var metadata) ? CloneMetadata(metadata) : null;
        }
    }

    /// <summary>
    /// Broadcast global training state to all workers
    /// </summary>
    public async Task BroadcastGlobalStateAsync(GlobalTrainingState state)
    {
        // Implementation will be added in integration spec
        // This is a placeholder for future RPC communication
        await Task.CompletedTask;
    }

    /// <summary>
    /// Manually trigger a rescaling operation
    /// </summary>
    public void TriggerRescaling(RescaleType type)
    {
        lock (_lock)
        {
            var oldTopology = CloneTopology(_currentTopology);
            var newTopology = CloneTopology(_currentTopology);

            var evt = new RescalingEvent
            {
                Type = type,
                OldTopology = oldTopology,
                NewTopology = newTopology,
                TriggerReason = "Manual trigger"
            };

            RescalingTriggered?.Invoke(evt);
        }
    }

    /// <summary>
    /// Get the current global training state
    /// </summary>
    public GlobalTrainingState GetGlobalState(int currentEpoch, int currentStep, float learningRate)
    {
        lock (_lock)
        {
            return new GlobalTrainingState
            {
                CurrentEpoch = currentEpoch,
                CurrentStep = currentStep,
                LearningRate = learningRate,
                ActiveWorkerCount = _currentTopology.WorldSize,
                StateTimestamp = DateTime.UtcNow
            };
        }
    }

    private void StartStabilityTimer()
    {
        _stabilityTimer?.Dispose();

        _stabilityTimer = new Timer(
            _ => CheckTopologyStability(),
            null,
            _config.StabilityWindowMs,
            Timeout.Infinite
        );
    }

    private void StartHealthCheckTimer()
    {
        _healthCheckTimer = new Timer(
            _ => PerformHealthCheck(),
            null,
            _config.WorkerHeartbeatTimeoutMs / 2,
            _config.WorkerHeartbeatTimeoutMs / 2
        );
    }

    private void CheckTopologyStability()
    {
        lock (_lock)
        {
            if (_pendingWorkers == null || _pendingWorkers.Count == 0)
            {
                return;
            }

            var oldTopology = CloneTopology(_currentTopology);

            foreach (var worker in _pendingWorkers)
            {
                if (_workers.TryGetValue(worker, out var metadata))
                {
                    metadata.Status = WorkerStatus.Active;
                    metadata.Rank = _currentTopology.WorldSize;
                    _currentTopology.AddWorker(worker);
                    WorkerStatusChanged?.Invoke(worker, WorkerStatus.Active);
                }
            }

            _pendingWorkers.Clear();

            if (_currentTopology.Workers.Count > oldTopology.Workers.Count)
            {
                TriggerRescalingIfStable(oldTopology);
            }
        }
    }

    private void PerformHealthCheck()
    {
        lock (_lock)
        {
            var timeout = TimeSpan.FromMilliseconds(_config.WorkerHeartbeatTimeoutMs);
            var failedWorkers = new List<WorkerId>();

            foreach (var (workerId, metadata) in _workers)
            {
                if (metadata.Status == WorkerStatus.Active && !metadata.IsHealthy(timeout))
                {
                    failedWorkers.Add(workerId);
                }
            }

            foreach (var worker in failedWorkers)
            {
                UnregisterWorker(worker, graceful: false);
            }
        }
    }

    private void TriggerRescalingIfStable(ClusterTopology oldTopology)
    {
        _lastTopologyChange = DateTime.UtcNow;

        var stabilityTimer = new Timer(
            _ =>
            {
                lock (_lock)
                {
                    if (_lastTopologyChange.HasValue &&
                        DateTime.UtcNow - _lastTopologyChange.Value >= TimeSpan.FromMilliseconds(_config.StabilityWindowMs))
                    {
                        var newTopology = CloneTopology(_currentTopology);

                        if (newTopology.WorldSize != oldTopology.WorldSize)
                        {
                            var addedWorkers = newTopology.Workers.Except(oldTopology.Workers).ToList();
                            var removedWorkers = oldTopology.Workers.Except(newTopology.Workers).ToList();

                            var evt = new RescalingEvent
                            {
                                Type = newTopology.WorldSize > oldTopology.WorldSize ? RescaleType.ScaleUp : RescaleType.ScaleDown,
                                OldTopology = oldTopology,
                                NewTopology = newTopology,
                                AddedWorkers = addedWorkers,
                                RemovedWorkers = removedWorkers,
                                TriggerReason = "Topology change detected"
                            };

                            RescalingTriggered?.Invoke(evt);
                        }

                        _lastTopologyChange = null;
                    }
                }
            },
            null,
            _config.StabilityWindowMs,
            Timeout.Infinite
        );
    }

    private void CheckFailureTolerance()
    {
        var toleranceThreshold = (int)Math.Ceiling(_config.MaxWorkers * _config.FailureTolerancePercentage / 100.0);

        if (_consecutiveFailures >= _config.MaxConsecutiveFailures ||
            _currentTopology.WorldSize < _config.MinWorkers)
        {
            throw new InvalidOperationException("Failure tolerance exceeded. Training must be aborted.");
        }
    }

    private ClusterTopology CloneTopology(ClusterTopology topology)
    {
        return new ClusterTopology
        {
            WorldSize = topology.WorldSize,
            Workers = new List<WorkerId>(topology.Workers),
            LastUpdated = topology.LastUpdated,
            Epoch = topology.Epoch
        };
    }

    private WorkerMetadata CloneMetadata(WorkerMetadata metadata)
    {
        return new WorkerMetadata
        {
            WorkerId = metadata.WorkerId,
            Status = metadata.Status,
            JoinTime = metadata.JoinTime,
            LastHeartbeat = metadata.LastHeartbeat,
            Rank = metadata.Rank,
            LocalWorldSize = metadata.LocalWorldSize,
            CustomAttributes = new Dictionary<string, string>(metadata.CustomAttributes)
        };
    }

    public void Dispose()
    {
        _healthCheckTimer?.Dispose();
        _stabilityTimer?.Dispose();
        GC.SuppressFinalize(this);
    }
}
```

## Implementation Notes

1. Thread Safety: All public methods use lock statements to ensure thread-safe access to shared state
2. Health Checking: Periodic timer checks worker heartbeats and marks unresponsive workers as failed
3. Stability Window: Rescaling is only triggered after a configurable stability period to handle rapid topology fluctuations
4. Failure Tolerance: Tracks consecutive failures and aborts if tolerance is exceeded
5. Event-Driven: Uses C# events to notify subscribers of rescaling events and worker status changes

## Dependencies
- Configuration and data models from spec_elastic_config_models.md
- System.Timers for periodic health checks

## Estimated Effort
~50 minutes

## Success Criteria
- Worker registration and unregistration work correctly
- Health check mechanism identifies failed workers based on heartbeat timeout
- Rescaling events are triggered with proper stability window delay
- Failure tolerance is enforced correctly
- Thread safety is maintained under concurrent access
- All public events are fired with appropriate data
