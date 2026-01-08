namespace MachineLearning.Checkpointing;

/// <summary>
/// Validator for checkpoint integrity and compatibility
/// </summary>
public class CheckpointValidator : ICheckpointValidator
{
    private readonly ICheckpointStorage _storage;
    private readonly List<IIntegrityChecker> _integrityCheckers;
    private readonly List<ICompatibilityChecker> _compatibilityCheckers;

    /// <summary>
    /// Create a new CheckpointValidator
    /// </summary>
    public CheckpointValidator(
        ICheckpointStorage storage,
        IEnumerable<IIntegrityChecker>? integrityCheckers = null,
        IEnumerable<ICompatibilityChecker>? compatibilityCheckers = null)
    {
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
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

    /// <summary>
    /// Validate a checkpoint
    /// </summary>
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

    /// <summary>
    /// Validate checkpoint metadata
    /// </summary>
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

        return await Task.FromResult(result);
    }

    /// <summary>
    /// Validate checkpoint shards
    /// </summary>
    public async Task<ValidationResult> ValidateShardsAsync(
        CheckpointMetadata metadata,
        string checkpointPrefix,
        CancellationToken cancellationToken = default)
    {
        var result = new ValidationResult();

        foreach (var shardMeta in metadata.Shards ?? new List<ShardMetadata>())
        {
            var shardPath = $"{checkpointPrefix}_shard_{shardMeta.Rank}.shard";
            var shardResult = await ValidateShardAsync(shardPath, shardMeta, cancellationToken);
            result.AddSubResults(shardResult);
        }

        return await Task.FromResult(result);
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

        return await Task.FromResult(result);
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

        return await Task.FromResult(result);
    }

    private async Task<CheckpointMetadata> LoadMetadataAsync(
        string prefix,
        CancellationToken cancellationToken)
    {
        var metadataPath = $"{prefix}.metadata.json";
        var data = await _storage.ReadAsync(metadataPath, cancellationToken);
        var metadataJson = System.Text.Encoding.UTF8.GetString(data);
        return MetadataSerializer.Deserialize(metadataJson);
    }
}
