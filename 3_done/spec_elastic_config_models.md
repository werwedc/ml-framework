# Spec: Elastic Training Configuration and Data Models

## Overview
Define the foundational configuration classes, enums, and data transfer objects (DTOs) that enable elastic training functionality. This includes configuration options, worker metadata, cluster state representation, and training state synchronization models.

## Deliverables

### 1. Enums

**File:** `src/MachineLearning/Distributed/Enums/AdaptationStrategy.cs`
```csharp
namespace MachineLearning.Distributed.Enums;

/// <summary>
/// Learning rate adaptation strategies when worker count changes
/// </summary>
public enum AdaptationStrategy
{
    /// <summary>
    /// Scale LR proportionally to new worker count
    /// </summary>
    Linear,

    /// <summary>
    /// LR scales with sqrt(worker_count) for more stability
    /// </summary>
    SquareRoot,

    /// <summary>
    /// Keep global LR constant, only change throughput
    /// </summary>
    None
}
```

**File:** `src/MachineLearning/Distributed/Enums/RedistributionType.cs`
```csharp
namespace MachineLearning.Distributed.Enums;

/// <summary>
/// Data redistribution strategies when topology changes
/// </summary>
public enum RedistributionType
{
    /// <summary>
    /// Redistribute all data across new worker set (better load balance, more data movement)
    /// </summary>
    FullReshuffle,

    /// <summary>
    /// Keep existing workers' data, only redistribute from lost/new workers (faster, temporary imbalance)
    /// </summary>
    Incremental
}
```

**File:** `src/MachineLearning/Distributed/Enums/RescaleType.cs`
```csharp
namespace MachineLearning.Distributed.Enums;

/// <summary>
/// Types of rescaling operations
/// </summary>
public enum RescaleType
{
    /// <summary>
    /// Adding new workers to the cluster
    /// </summary>
    ScaleUp,

    /// <summary>
    /// Removing workers from the cluster
    /// </summary>
    ScaleDown,

    /// <summary>
    /// Replacement of failed workers
    /// </summary>
    Replace
}
```

**File:** `src/MachineLearning/Distributed/Enums/WorkerStatus.cs`
```csharp
namespace MachineLearning.Distributed.Enums;

/// <summary>
/// Status of a worker in the elastic cluster
/// </summary>
public enum WorkerStatus
{
    /// <summary>
    /// Worker has just joined and is initializing
    /// </summary>
    Joining,

    /// <summary>
    /// Worker is actively training
    /// </summary>
    Active,

    /// <summary>
    /// Worker is in rescaling process
    /// </summary>
    Rescaling,

    /// <summary>
    /// Worker is gracefully shutting down
    /// </summary>
    Leaving,

    /// <summary>
    /// Worker has failed or become unresponsive
    /// </summary>
    Failed
}
```

### 2. Data Models

**File:** `src/MachineLearning/Distributed/Models/WorkerId.cs`
```csharp
namespace MachineLearning.Distributed.Models;

/// <summary>
/// Unique identifier for a worker in the cluster
/// </summary>
public record WorkerId
{
    public string Id { get; init; } = string.Empty;
    public string Hostname { get; init; } = string.Empty;
    public int Port { get; init; }

    public WorkerId(string id, string hostname, int port)
    {
        Id = id;
        Hostname = hostname;
        Port = port;
    }

    public override string ToString() => $"{Id}@{Hostname}:{Port}";
}
```

**File:** `src/MachineLearning/Distributed/Models/WorkerMetadata.cs`
```csharp
namespace MachineLearning.Distributed.Models;

using MachineLearning.Distributed.Enums;

/// <summary>
/// Metadata information about a worker
/// </summary>
public class WorkerMetadata
{
    public WorkerId WorkerId { get; set; } = null!;
    public WorkerStatus Status { get; set; }
    public DateTime JoinTime { get; set; }
    public DateTime LastHeartbeat { get; set; }
    public int Rank { get; set; }
    public int LocalWorldSize { get; set; }
    public Dictionary<string, string> CustomAttributes { get; set; } = new();

    public bool IsHealthy(TimeSpan timeout)
    {
        return DateTime.UtcNow - LastHeartbeat < timeout;
    }
}
```

**File:** `src/MachineLearning/Distributed/Models/ClusterTopology.cs`
```csharp
namespace MachineLearning.Distributed.Models;

/// <summary>
/// Represents the current cluster topology
/// </summary>
public class ClusterTopology
{
    public int WorldSize { get; set; }
    public List<WorkerId> Workers { get; set; } = new();
    public DateTime LastUpdated { get; set; }
    public int Epoch { get; set; }

    public ClusterTopology()
    {
        LastUpdated = DateTime.UtcNow;
    }

    public void AddWorker(WorkerId worker)
    {
        Workers.Add(worker);
        WorldSize = Workers.Count;
        LastUpdated = DateTime.UtcNow;
        Epoch++;
    }

    public void RemoveWorker(WorkerId worker)
    {
        Workers.Remove(worker);
        WorldSize = Workers.Count;
        LastUpdated = DateTime.UtcNow;
        Epoch++;
    }
}
```

**File:** `src/MachineLearning/Distributed/Models/GlobalTrainingState.cs`
```csharp
namespace MachineLearning.Distributed.Models;

/// <summary>
/// Global training state that needs to be synchronized across workers
/// </summary>
public class GlobalTrainingState
{
    public int CurrentEpoch { get; set; }
    public int CurrentStep { get; set; }
    public float LearningRate { get; set; }
    public int GlobalBatchSize { get; set; }
    public int ActiveWorkerCount { get; set; }
    public DateTime StateTimestamp { get; set; }

    // Optional: serialized optimizer state
    public byte[]? OptimizerState { get; set; }

    public GlobalTrainingState()
    {
        StateTimestamp = DateTime.UtcNow;
    }
}
```

**File:** `src/MachineLearning/Distributed/Models/RescalingEvent.cs`
```csharp
namespace MachineLearning.Distributed.Models;

using MachineLearning.Distributed.Enums;

/// <summary>
/// Event representing a topology change requiring rescaling
/// </summary>
public class RescalingEvent
{
    public RescaleType Type { get; set; }
    public ClusterTopology OldTopology { get; set; } = null!;
    public ClusterTopology NewTopology { get; set; } = null!;
    public List<WorkerId> AddedWorkers { get; set; } = new();
    public List<WorkerId> RemovedWorkers { get; set; } = new();
    public DateTime EventTime { get; set; }
    public string TriggerReason { get; set; } = string.Empty;

    public RescalingEvent()
    {
        EventTime = DateTime.UtcNow;
    }
}
```

**File:** `src/MachineLearning/Distributed/Models/DataShard.cs`
```csharp
namespace MachineLearning.Distributed.Models;

/// <summary>
/// Represents a shard of training data
/// </summary>
public record DataShard
{
    public int ShardId { get; init; }
    public int StartIndex { get; init; }
    public int EndIndex { get; init; }
    public int Size => EndIndex - StartIndex;

    public DataShard(int shardId, int startIndex, int endIndex)
    {
        ShardId = shardId;
        StartIndex = startIndex;
        EndIndex = endIndex;
    }
}
```

### 3. Configuration Class

**File:** `src/MachineLearning/Distributed/Configuration/ElasticTrainingConfig.cs`
```csharp
namespace MachineLearning.Distributed.Configuration;

using MachineLearning.Distributed.Enums;

/// <summary>
/// Configuration for elastic training behavior
/// </summary>
public class ElasticTrainingConfig
{
    /// <summary>
    /// Minimum and maximum worker count
    /// </summary>
    public int MinWorkers { get; set; } = 1;
    public int MaxWorkers { get; set; } = int.MaxValue;

    /// <summary>
    /// How long to wait for new workers before proceeding (milliseconds)
    /// </summary>
    public int RescaleTimeoutMs { get; set; } = 30000;

    /// <summary>
    /// Maximum number of consecutive failures before aborting
    /// </summary>
    public int MaxConsecutiveFailures { get; set; } = 3;

    /// <summary>
    /// Learning rate adaptation strategy
    /// </summary>
    public AdaptationStrategy LRAdaptationStrategy { get; set; } = AdaptationStrategy.Linear;

    /// <summary>
    /// Data redistribution method
    /// </summary>
    public RedistributionType RedistributionType { get; set; } = RedistributionType.FullReshuffle;

    /// <summary>
    /// Whether to use synchronous or asynchronous rescaling
    /// </summary>
    public bool UseSynchronousRescaling { get; set; } = true;

    /// <summary>
    /// Stability window before triggering rescaling (milliseconds)
    /// </summary>
    public int StabilityWindowMs { get; set; } = 30000;

    /// <summary>
    /// Timeout for worker heartbeats before considering it failed (milliseconds)
    /// </summary>
    public int WorkerHeartbeatTimeoutMs { get; set; } = 10000;

    /// <summary>
    /// Number of worker failures to tolerate before aborting (as percentage)
    /// </summary>
    public int FailureTolerancePercentage { get; set; } = 20;

    /// <summary>
    /// Whether to use parallel data transfer during redistribution
    /// </summary>
    public bool UseParallelDataTransfer { get; set; } = true;

    public void Validate()
    {
        if (MinWorkers < 1)
            throw new ArgumentException("MinWorkers must be at least 1");

        if (MaxWorkers < MinWorkers)
            throw new ArgumentException("MaxWorkers must be >= MinWorkers");

        if (RescaleTimeoutMs < 0)
            throw new ArgumentException("RescaleTimeoutMs cannot be negative");

        if (MaxConsecutiveFailures < 0)
            throw new ArgumentException("MaxConsecutiveFailures cannot be negative");

        if (StabilityWindowMs < 0)
            throw new ArgumentException("StabilityWindowMs cannot be negative");

        if (WorkerHeartbeatTimeoutMs <= 0)
            throw new ArgumentException("WorkerHeartbeatTimeoutMs must be positive");

        if (FailureTolerancePercentage < 0 || FailureTolerancePercentage > 100)
            throw new ArgumentException("FailureTolerancePercentage must be between 0 and 100");
    }
}
```

## Implementation Notes

1. All classes should be placed in the appropriate namespaces under `src/MachineLearning/Distributed/`
2. Use record types for immutable data structures (WorkerId, DataShard)
3. Include XML documentation comments for all public members
4. Implement validation logic in ElasticTrainingConfig.Validate()
5. Ensure thread-safety for ClusterTopology if concurrent access is expected

## Dependencies
- None (these are foundational data models)

## Estimated Effort
~45 minutes

## Success Criteria
- All enums defined with appropriate values
- All data models compile without errors
- Configuration validation logic works correctly
- XML documentation is complete for all public members
