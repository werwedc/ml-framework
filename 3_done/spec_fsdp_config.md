# Spec: FSDP Configuration

## Overview
Define configuration classes and enums for FSDP (Fully Sharded Data Parallelism) settings.

## Requirements

### 1. ShardingStrategy Enum
Define an enum for different sharding strategies:
- `Full`: Shard all parameters across all devices (maximum memory savings)
- `LayerWise`: Shard individual layers sequentially (better for communication)
- `Hybrid`: Mix of full and layer-wise for optimal performance

```csharp
public enum ShardingStrategy
{
    /// <summary>Shard all parameters across all devices</summary>
    Full,

    /// <summary>Shard individual layers sequentially</summary>
    LayerWise,

    /// <summary>Mix of full and layer-wise sharding</summary>
    Hybrid
}
```

### 2. FSDPConfig Class
Create a configuration class to hold all FSDP settings:

```csharp
public class FSDPConfig
{
    /// <summary>Sharding strategy to use</summary>
    public ShardingStrategy ShardingStrategy { get; set; } = ShardingStrategy.Full;

    /// <summary>Enable mixed precision (FP16/BF16)</summary>
    public bool MixedPrecision { get; set; } = true;

    /// <summary>Offload parameters/gradients to CPU when not in use</summary>
    public bool OffloadToCPU { get; set; } = false;

    /// <summary>Enable activation checkpointing</summary>
    public bool ActivationCheckpointing { get; set; } = false;

    /// <summary>Bucket size for gradient communication (in MB)</summary>
    public int BucketSizeMB { get; set; } = 25;

    /// <summary>Number of communication workers</summary>
    public int NumCommunicationWorkers { get; set; } = 2;

    /// <summary>Validate configuration</summary>
    public void Validate()
    {
        if (BucketSizeMB <= 0 || BucketSizeMB > 1000)
        {
            throw new ArgumentException("BucketSizeMB must be between 1 and 1000", nameof(BucketSizeMB));
        }

        if (NumCommunicationWorkers <= 0 || NumCommunicationWorkers > 16)
        {
            throw new ArgumentException("NumCommunicationWorkers must be between 1 and 16", nameof(NumCommunicationWorkers));
        }
    }
}
```

### 3. FSDPState Class
Create a class to track the state of sharded parameters:

```csharp
public class FSDPState
{
    /// <summary>Owner rank of this parameter shard</summary>
    public int OwnerRank { get; set; }

    /// <summary>Number of shards across all devices</summary>
    public int NumShards { get; set; }

    /// <summary>Local shard index</summary>
    public int ShardIndex { get; set; }

    /// <summary>Whether this shard is currently gathered on device</summary>
    public bool IsGathered { get; set; }

    /// <summary>Whether this shard is currently offloaded to CPU</summary>
    public bool IsOffloaded { get; set; }
}
```

## Directory Structure
- **File**: `src/MLFramework/Distributed/FSDP/FSDPConfig.cs`
- **Namespace**: `MLFramework.Distributed.FSDP`

## Dependencies
- None (standalone configuration classes)

## Implementation Notes
1. All classes should be in the `MLFramework.Distributed.FSDP` namespace
2. Use XML documentation comments for all public members
3. Include validation logic in `FSDPConfig.Validate()`
4. Use C# default properties with sensible defaults

## Testing Requirements
- Test validation logic with invalid inputs
- Test default configuration values
- Test serialization/deserialization (optional)

## Estimated Time
30 minutes
