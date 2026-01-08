namespace MachineLearning.Checkpointing;

/// <summary>
/// Result of metadata validation
/// </summary>
public class ValidationResult
{
    /// <summary>
    /// Whether the validation passed
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// List of validation errors
    /// </summary>
    public List<string> Errors { get; set; } = new List<string>();

    /// <summary>
    /// Create a successful validation result
    /// </summary>
    public static ValidationResult Success() => new ValidationResult { IsValid = true };

    /// <summary>
    /// Create a failed validation result with errors
    /// </summary>
    public static ValidationResult Failure(params string[] errors) =>
        new ValidationResult { IsValid = false, Errors = errors.ToList() };
}

/// <summary>
/// Validator for checkpoint metadata
/// </summary>
public static class MetadataValidator
{
    /// <summary>
    /// Validate checkpoint metadata
    /// </summary>
    public static ValidationResult Validate(CheckpointMetadata metadata)
    {
        if (metadata == null)
            return ValidationResult.Failure("Metadata cannot be null");

        var errors = new List<string>();

        // Validate version
        if (string.IsNullOrWhiteSpace(metadata.Version))
            errors.Add("Version is required");

        // Validate timestamp
        if (metadata.Timestamp == default)
            errors.Add("Timestamp is required");

        // Validate world size
        if (metadata.WorldSize <= 0)
            errors.Add("World size must be positive");

        // Validate DDP rank
        if (metadata.DdpRank < 0 || metadata.DdpRank >= metadata.WorldSize)
            errors.Add($"DDP rank must be between 0 and {metadata.WorldSize - 1}");

        // Validate sharding
        if (metadata.Sharding != null)
        {
            if (string.IsNullOrWhiteSpace(metadata.Sharding.Strategy))
                errors.Add("Sharding strategy is required");

            if (metadata.Sharding.ShardCount <= 0)
                errors.Add("Shard count must be positive");

            if (string.IsNullOrWhiteSpace(metadata.Sharding.Precision))
                errors.Add("Precision is required");
        }

        // Validate shards
        if (metadata.Shards != null && metadata.Shards.Count > 0)
        {
            var shardRanks = new HashSet<int>();
            foreach (var shard in metadata.Shards)
            {
                if (shard.Rank < 0 || shard.Rank >= metadata.WorldSize)
                    errors.Add($"Shard rank {shard.Rank} is out of range [0, {metadata.WorldSize - 1}]");

                if (shardRanks.Contains(shard.Rank))
                    errors.Add($"Duplicate shard rank {shard.Rank}");

                shardRanks.Add(shard.Rank);

                if (string.IsNullOrWhiteSpace(shard.FilePath))
                    errors.Add($"Shard {shard.Rank} must have a file path");

                if (shard.FileSize < 0)
                    errors.Add($"Shard {shard.Rank} has invalid file size");
            }

            // Validate shard count matches
            if (metadata.Sharding != null && metadata.Shards.Count != metadata.Sharding.ShardCount)
                errors.Add($"Shard count mismatch: metadata specifies {metadata.Sharding.ShardCount} but {metadata.Shards.Count} shards found");
        }

        return errors.Count == 0 ? ValidationResult.Success() : ValidationResult.Failure(errors.ToArray());
    }
}
