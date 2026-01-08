# Spec: Checkpoint Metadata Format

## Overview
Define the metadata structure that tracks sharding scheme, file locations, version compatibility, and validation information for distributed checkpoints.

## Scope
- 30-45 minutes coding time
- Focus on data structures and serialization
- Target: `src/MLFramework/Checkpointing/Metadata/`

## Classes

### 1. CheckpointMetadata (Main Data Structure)
```csharp
public class CheckpointMetadata
{
    public string Version { get; set; } = "1.0.0";
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public int WorldSize { get; set; } // Number of GPUs that saved the checkpoint
    public int DdpRank { get; set; } // Rank that created this metadata
    public ShardingMetadata Sharding { get; set; }
    public List<ShardMetadata> Shards { get; set; }
    public TrainingMetadata Training { get; set; }
    public Dictionary<string, string> CustomFields { get; set; }
}
```

### 2. ShardingMetadata (Sharding Scheme Info)
```csharp
public class ShardingMetadata
{
    public string Strategy { get; set; } // "fsdp", "ddp", "tensor_parallel"
    public int ShardCount { get; set; }
    public string Precision { get; set; } // "fp16", "bf16", "fp32"
    public Dictionary<string, object> StrategySpecificInfo { get; set; }
}
```

### 3. ShardMetadata (Per-Shard Information)
```csharp
public class ShardMetadata
{
    public int Rank { get; set; }
    public string FilePath { get; set; }
    public long FileSize { get; set; }
    public List<TensorMetadata> Tensors { get; set; }
    public string Checksum { get; set; } // SHA-256 for integrity
}
```

### 4. TensorMetadata (Per-Tensor Information)
```csharp
public class TensorMetadata
{
    public string Name { get; set; }
    public long[] Shape { get; set; }
    public string DataType { get; set; }
    public long Offset { get; set; } // Byte offset in shard file
    public long Size { get; set; } // Size in bytes
}
```

### 5. TrainingMetadata (Training State)
```csharp
public class TrainingMetadata
{
    public int Epoch { get; set; }
    public long Step { get; set; }
    public float LearningRate { get; set; }
    public string OptimizerType { get; set; }
    public Dictionary<string, object> OptimizerState { get; set; }
}
```

### 6. MetadataSerializer (Serialization/Deserialization)
```csharp
public class MetadataSerializer
{
    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    public static string Serialize(CheckpointMetadata metadata)
    {
        return JsonSerializer.Serialize(metadata, Options);
    }

    public static CheckpointMetadata Deserialize(string json)
    {
        return JsonSerializer.Deserialize<CheckpointMetadata>(json, Options)
            ?? throw new InvalidOperationException("Failed to deserialize metadata");
    }

    public static async Task WriteAsync(
        CheckpointMetadata metadata,
        string path,
        CancellationToken cancellationToken = default)
    {
        var json = Serialize(metadata);
        await File.WriteAllTextAsync(path, json, cancellationToken);
    }

    public static async Task<CheckpointMetadata> ReadAsync(
        string path,
        CancellationToken cancellationToken = default)
    {
        var json = await File.ReadAllTextAsync(path, cancellationToken);
        return Deserialize(json);
    }
}
```

### 7. MetadataValidator (Validation Logic)
```csharp
public class MetadataValidator
{
    public static ValidationResult Validate(CheckpointMetadata metadata)
    {
        var result = new ValidationResult();

        // Version check
        if (string.IsNullOrEmpty(metadata.Version))
        {
            result.AddError("Version is required");
        }

        // Sharding info check
        if (metadata.Sharding == null)
        {
            result.AddError("Sharding metadata is required");
        }

        // Shard consistency check
        if (metadata.Shards == null || metadata.Shards.Count == 0)
        {
            result.AddError("At least one shard is required");
        }
        else if (metadata.Shards.Count != metadata.Sharding.ShardCount)
        {
            result.AddError($"Shard count mismatch: expected {metadata.Sharding.ShardCount}, found {metadata.Shards.Count}");
        }

        // Checksum integrity check (optional - can be deferred)
        foreach (var shard in metadata.Shards)
        {
            if (string.IsNullOrEmpty(shard.Checksum))
            {
                result.AddWarning($"Shard {shard.Rank} missing checksum");
            }
        }

        return result;
    }
}
```

### 8. ValidationResult (Validation Output)
```csharp
public class ValidationResult
{
    public List<string> Errors { get; } = new();
    public List<string> Warnings { get; } = new();

    public bool IsValid => Errors.Count == 0;

    public void AddError(string error) => Errors.Add(error);
    public void AddWarning(string warning) => Warnings.Add(warning);
}
```

## File Naming Convention
- Metadata file: `{checkpoint_prefix}.metadata.json`

## Integration Points
- Created by: `DistributedCheckpointCoordinator`
- Stored via: `ICheckpointStorage`
- Read by: `DistributedCheckpointLoader`

## Testing Requirements
- Test serialization roundtrip
- Test validation with valid metadata
- Test validation with missing fields
- Test validation with inconsistent shard counts
- Test custom fields persistence

## Success Criteria
- Can serialize and deserialize all metadata fields
- Validation catches common errors
- Version information is preserved
- Supports extensible custom fields
