# Spec: Batching Configuration

## Overview
Define a configuration class to manage dynamic batching parameters for model serving.

## Technical Requirements

### Class Definition
```csharp
namespace MLFramework.Serving;

/// <summary>
/// Configuration parameters for dynamic batching behavior
/// </summary>
public class BatchingConfiguration
{
    /// <summary>
    /// Maximum number of requests per batch
    /// </summary>
    public int MaxBatchSize { get; set; }

    /// <summary>
    /// Maximum time to wait before dispatching incomplete batch
    /// </summary>
    public TimeSpan MaxWaitTime { get; set; }

    /// <summary>
    /// Target batch size for optimal GPU utilization
    /// </summary>
    public int PreferBatchSize { get; set; }

    /// <summary>
    /// Maximum queue size to prevent memory exhaustion
    /// </summary>
    public int MaxQueueSize { get; set; }

    /// <summary>
    /// Strategy for handling timeout scenarios
    /// </summary>
    public TimeoutStrategy TimeoutStrategy { get; set; }
}

/// <summary>
/// Defines behavior when timeout expires before batch is full
/// </summary>
public enum TimeoutStrategy
{
    /// <summary>
    /// Dispatch partial batch immediately
    /// </summary>
    DispatchPartial,

    /// <summary>
    /// Wait additional time for full batch
    /// </summary>
    WaitForFull,

    /// <summary>
    /// Hybrid: dispatch partial if queue not growing rapidly
    /// </summary>
    Adaptive
}
```

### Validation Rules
1. `MaxBatchSize` must be between 1 and 1024
2. `MaxWaitTime` must be between 1ms and 1000ms
3. `PreferBatchSize` must be between 1 and `MaxBatchSize`
4. `MaxQueueSize` must be between 10 and 10000

### Methods
```csharp
public class BatchingConfiguration
{
    /// <summary>
    /// Validate configuration and throw exceptions for invalid settings
    /// </summary>
    public void Validate()
    {
        if (MaxBatchSize < 1 || MaxBatchSize > 1024)
            throw new ArgumentOutOfRangeException(nameof(MaxBatchSize));

        if (MaxWaitTime < TimeSpan.FromMilliseconds(1) || MaxWaitTime > TimeSpan.FromMilliseconds(1000))
            throw new ArgumentOutOfRangeException(nameof(MaxWaitTime));

        if (PreferBatchSize < 1 || PreferBatchSize > MaxBatchSize)
            throw new ArgumentOutOfRangeException(nameof(PreferBatchSize));

        if (MaxQueueSize < 10 || MaxQueueSize > 10000)
            throw new ArgumentOutOfRangeException(nameof(MaxQueueSize));
    }

    /// <summary>
    /// Create default configuration for typical model serving
    /// </summary>
    public static BatchingConfiguration Default()
    {
        return new BatchingConfiguration
        {
            MaxBatchSize = 32,
            MaxWaitTime = TimeSpan.FromMilliseconds(5),
            PreferBatchSize = 16,
            MaxQueueSize = 100,
            TimeoutStrategy = TimeoutStrategy.DispatchPartial
        };
    }
}
```

## File Location
- **Path:** `src/Serving/BatchingConfiguration.cs`

## Dependencies
- None

## Success Criteria
- Configuration class compiles successfully
- All validation rules throw appropriate exceptions
- Default configuration provides sensible defaults
- All properties are properly documented

## Testing Requirements
- Test validation with invalid MaxBatchSize
- Test validation with invalid MaxWaitTime
- Test validation with invalid PreferBatchSize relative to MaxBatchSize
- Test validation with invalid MaxQueueSize
- Test default configuration creation
- Test all enum values
