# Spec: Checkpoint Validation and Integrity

## Overview
Implement comprehensive checkpoint validation including integrity checks, version compatibility verification, and rollback capabilities to ensure checkpoints are valid and safe to load.

## Scope
- 30-45 minutes coding time
- Focus on validation logic and integrity checks
- Target: `src/MLFramework/Checkpointing/Validation/`

## Classes

### 1. ICheckpointValidator (Interface)
```csharp
public interface ICheckpointValidator
{
    Task<ValidationResult> ValidateCheckpointAsync(
        string checkpointPath,
        CancellationToken cancellationToken = default);

    Task<ValidationResult> ValidateMetadataAsync(
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default);

    Task<ValidationResult> ValidateShardsAsync(
        CheckpointMetadata metadata,
        string checkpointPrefix,
        CancellationToken cancellationToken = default);
}
```

### 2. CheckpointValidator (Main Validator)
```csharp
public class CheckpointValidator : ICheckpointValidator
{
    private readonly ICheckpointStorage _storage;
    private readonly List<IIntegrityChecker> _integrityCheckers;
    private readonly List<ICompatibilityChecker> _compatibilityCheckers;

    public CheckpointValidator(
        ICheckpointStorage storage,
        IEnumerable<IIntegrityChecker>? integrityCheckers = null,
        IEnumerable<ICompatibilityChecker>? compatibilityCheckers = null)
    {
        _storage = storage;
        _integrityCheckers = integrityCheckers?.ToList() ?? new List<IIntegrityChecker>
        {
            new ChecksumIntegrityChecker(),
            new SizeIntegrityChecker()
        };
        _compatibilityCheckers = compatibilityCheckers?.ToList() ?? new List<ICompatibilityChecker>
        {
            new VersionCompatibilityChecker(),
            new SchemaCompatibilityChecker()
        };
    }

    public async Task<ValidationResult> ValidateCheckpointAsync(
        string checkpointPath,
        CancellationToken cancellationToken = default)
    {
        var result = new ValidationResult();

        try
        {
            // Check if checkpoint exists
            var exists = await _storage.ExistsAsync(checkpointPath, cancellationToken);
            if (!exists)
            {
                result.AddError($"Checkpoint not found: {checkpointPath}");
                return result;
            }

            // Determine checkpoint type (single-file or multi-shard)
            var isMultiShard = checkpointPath.EndsWith(".metadata.json");

            if (isMultiShard)
            {
                // Multi-shard checkpoint
                var prefix = checkpointPath.Replace(".metadata.json", "");
                var metadata = await LoadMetadataAsync(prefix, cancellationToken);
                result.AddSubResults(await ValidateMetadataAsync(metadata, cancellationToken));
                result.AddSubResults(await ValidateShardsAsync(metadata, prefix, cancellationToken));
            }
            else
            {
                // Single-file checkpoint
                result.AddSubResults(await ValidateSingleFileAsync(checkpointPath, cancellationToken));
            }
        }
        catch (Exception ex)
        {
            result.AddError($"Validation failed: {ex.Message}");
        }

        return result;
    }

    public async Task<ValidationResult> ValidateMetadataAsync(
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default)
    {
        var result = new ValidationResult();

        // Validate required fields
        if (string.IsNullOrEmpty(metadata.Version))
        {
            result.AddError("Metadata version is missing");
        }

        if (metadata.Sharding == null)
        {
            result.AddError("Sharding metadata is missing");
        }

        if (metadata.Shards == null || metadata.Shards.Count == 0)
        {
            result.AddError("No shards found in metadata");
        }

        // Validate consistency
        if (metadata.Sharding != null && metadata.Shards != null)
        {
            if (metadata.Shards.Count != metadata.Sharding.ShardCount)
            {
                result.AddError($"Shard count mismatch: metadata says {metadata.Sharding.ShardCount}, found {metadata.Shards.Count}");
            }

            // Check for duplicate ranks
            var ranks = metadata.Shards.Select(s => s.Rank).ToList();
            var duplicateRanks = ranks.GroupBy(r => r).Where(g => g.Count() > 1).Select(g => g.Key);
            foreach (var rank in duplicateRanks)
            {
                result.AddError($"Duplicate rank found: {rank}");
            }
        }

        // Run compatibility checkers
        foreach (var checker in _compatibilityCheckers)
        {
            var subResult = await checker.CheckCompatibilityAsync(metadata, cancellationToken);
            result.AddSubResults(subResult);
        }

        return result;
    }

    public async Task<ValidationResult> ValidateShardsAsync(
        CheckpointMetadata metadata,
        string checkpointPrefix,
        CancellationToken cancellationToken = default)
    {
        var result = new ValidationResult();

        foreach (var shardMeta in metadata.Shards)
        {
            var shardPath = $"{checkpointPrefix}_shard_{shardMeta.Rank}.shard";
            var shardResult = await ValidateShardAsync(shardPath, shardMeta, cancellationToken);
            result.AddSubResults(shardResult);
        }

        return result;
    }

    private async Task<ValidationResult> ValidateShardAsync(
        string shardPath,
        ShardMetadata shardMeta,
        CancellationToken cancellationToken)
    {
        var result = new ValidationResult();

        // Check if shard exists
        var exists = await _storage.ExistsAsync(shardPath, cancellationToken);
        if (!exists)
        {
            result.AddError($"Shard file not found: {shardPath}");
            return result;
        }

        // Read shard data
        byte[] shardData;
        try
        {
            shardData = await _storage.ReadAsync(shardPath, cancellationToken);
        }
        catch (Exception ex)
        {
            result.AddError($"Failed to read shard {shardPath}: {ex.Message}");
            return result;
        }

        // Run integrity checkers
        foreach (var checker in _integrityCheckers)
        {
            var subResult = await checker.CheckIntegrityAsync(
                shardData,
                shardMeta,
                cancellationToken);
            result.AddSubResults(subResult);
        }

        return result;
    }

    private async Task<ValidationResult> ValidateSingleFileAsync(
        string checkpointPath,
        CancellationToken cancellationToken)
    {
        var result = new ValidationResult();

        // Read checkpoint data
        byte[] data;
        try
        {
            data = await _storage.ReadAsync(checkpointPath, cancellationToken);
        }
        catch (Exception ex)
        {
            result.AddError($"Failed to read checkpoint: {ex.Message}");
            return result;
        }

        // Verify magic number
        if (data.Length < 4)
        {
            result.AddError("Checkpoint file is too small");
            return result;
        }

        var magic = BitConverter.ToInt32(data, 0);
        if (magic != 0x4D4C4350) // "MLCP"
        {
            result.AddError("Invalid checkpoint file: magic number mismatch");
            return result;
        }

        // TODO: More thorough single-file validation

        return result;
    }

    private async Task<CheckpointMetadata> LoadMetadataAsync(
        string prefix,
        CancellationToken cancellationToken)
    {
        var metadataPath = $"{prefix}.metadata.json";
        var data = await _storage.ReadAsync(metadataPath, cancellationToken);
        var metadataJson = Encoding.UTF8.GetString(data);
        return MetadataSerializer.Deserialize(metadataJson);
    }
}
```

### 3. IIntegrityChecker (Interface)
```csharp
public interface IIntegrityChecker
{
    string Name { get; }
    Task<ValidationResult> CheckIntegrityAsync(
        byte[] shardData,
        ShardMetadata shardMeta,
        CancellationToken cancellationToken);
}
```

### 4. ChecksumIntegrityChecker (Checksum Validation)
```csharp
public class ChecksumIntegrityChecker : IIntegrityChecker
{
    public string Name => "Checksum";

    public async Task<ValidationResult> CheckIntegrityAsync(
        byte[] shardData,
        ShardMetadata shardMeta,
        CancellationToken cancellationToken)
    {
        var result = new ValidationResult();

        if (string.IsNullOrEmpty(shardMeta.Checksum))
        {
            result.AddWarning($"Shard {shardMeta.Rank} has no checksum, skipping validation");
            return result;
        }

        var computedChecksum = ComputeChecksum(shardData);
        if (computedChecksum != shardMeta.Checksum)
        {
            result.AddError($"Shard {shardMeta.Rank} checksum mismatch: expected {shardMeta.Checksum}, computed {computedChecksum}");
        }

        return await Task.FromResult(result);
    }

    private string ComputeChecksum(byte[] data)
    {
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(data);
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
    }
}
```

### 5. SizeIntegrityChecker (Size Validation)
```csharp
public class SizeIntegrityChecker : IIntegrityChecker
{
    public string Name => "Size";

    public async Task<ValidationResult> CheckIntegrityAsync(
        byte[] shardData,
        ShardMetadata shardMeta,
        CancellationToken cancellationToken)
    {
        var result = new ValidationResult();

        if (shardData.Length != shardMeta.FileSize)
        {
            result.AddError($"Shard {shardMeta.Rank} size mismatch: expected {shardMeta.FileSize}, found {shardData.Length}");
        }

        return await Task.FromResult(result);
    }
}
```

### 6. ICompatibilityChecker (Interface)
```csharp
public interface ICompatibilityChecker
{
    string Name { get; }
    Task<ValidationResult> CheckCompatibilityAsync(
        CheckpointMetadata metadata,
        CancellationToken cancellationToken);
}
```

### 7. VersionCompatibilityChecker (Version Validation)
```csharp
public class VersionCompatibilityChecker : ICompatibilityChecker
{
    private readonly string _supportedVersionRange;

    public VersionCompatibilityChecker(string supportedVersionRange = "1.0.x")
    {
        _supportedVersionRange = supportedVersionRange;
    }

    public string Name => "Version";

    public async Task<ValidationResult> CheckCompatibilityAsync(
        CheckpointMetadata metadata,
        CancellationToken cancellationToken)
    {
        var result = new ValidationResult();

        try
        {
            var version = new Version(metadata.Version);
            var supportedVersion = new Version(_supportedVersionRange.Replace("x", "0"));

            if (version.Major != supportedVersion.Major)
            {
                result.AddError($"Incompatible checkpoint version: {metadata.Version} (supported: {_supportedVersionRange})");
            }
        }
        catch (Exception ex)
        {
            result.AddError($"Invalid version format: {ex.Message}");
        }

        return await Task.FromResult(result);
    }
}
```

### 8. SchemaCompatibilityChecker (Schema Validation)
```csharp
public class SchemaCompatibilityChecker : ICompatibilityChecker
{
    public string Name => "Schema";

    public async Task<ValidationResult> CheckCompatibilityAsync(
        CheckpointMetadata metadata,
        CancellationToken cancellationToken)
    {
        var result = new ValidationResult();

        // Validate sharding strategy
        if (metadata.Sharding != null)
        {
            var validStrategies = new[] { "fsdp", "ddp", "tensor_parallel" };
            if (!validStrategies.Contains(metadata.Sharding.Strategy.ToLower()))
            {
                result.AddWarning($"Unknown sharding strategy: {metadata.Sharding.Strategy}");
            }
        }

        // Validate precision
        if (metadata.Sharding != null && metadata.Sharding.Precision != null)
        {
            var validPrecisions = new[] { "fp16", "bf16", "fp32" };
            if (!validPrecisions.Contains(metadata.Sharding.Precision.ToLower()))
            {
                result.AddWarning($"Unknown precision: {metadata.Sharding.Precision}");
            }
        }

        return await Task.FromResult(result);
    }
}
```

### 9. ValidationResult (Validation Output)
```csharp
public class ValidationResult
{
    public List<string> Errors { get; } = new();
    public List<string> Warnings { get; } = new();
    public List<ValidationResult> SubResults { get; } = new();

    public bool IsValid => Errors.Count == 0 && SubResults.All(r => r.IsValid);
    public bool HasWarnings => Warnings.Count > 0 || SubResults.Any(r => r.HasWarnings);

    public void AddError(string error) => Errors.Add(error);
    public void AddWarning(string warning) => Warnings.Add(warning);
    public void AddSubResults(ValidationResult subResult)
    {
        SubResults.Add(subResult);
        foreach (var error in subResult.Errors)
        {
            Errors.Add($"{subResult.GetType().Name}: {error}");
        }
        foreach (var warning in subResult.Warnings)
        {
            Warnings.Add($"{subResult.GetType().Name}: {warning}");
        }
    }

    public string GetSummary()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Validation Result: {(IsValid ? "VALID" : "INVALID")}");

        if (Errors.Count > 0)
        {
            sb.AppendLine($"Errors ({Errors.Count}):");
            foreach (var error in Errors)
            {
                sb.AppendLine($"  - {error}");
            }
        }

        if (Warnings.Count > 0)
        {
            sb.AppendLine($"Warnings ({Warnings.Count}):");
            foreach (var warning in Warnings)
            {
                sb.AppendLine($"  - {warning}");
            }
        }

        return sb.ToString();
    }
}
```

## Integration Points
- Used by: `CheckpointLoader`, `DistributedCheckpoint.LoadAsync()`
- Depends on: `ICheckpointStorage`, `CheckpointMetadata`, `ShardMetadata`

## Testing Requirements
- Test metadata validation with valid and invalid data
- Test shard file validation
- Test checksum verification
- Test size verification
- Test version compatibility checking
- Test schema compatibility checking
- Test validation error aggregation

## Success Criteria
- Catches common checkpoint errors
- Provides clear validation messages
- Supports pluggable checkers
- Validates both single-file and multi-shard formats
- Returns detailed validation results
