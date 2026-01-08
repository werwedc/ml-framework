namespace MachineLearning.Checkpointing;

/// <summary>
/// Result of metadata validation
/// </summary>
public class ValidationResult
{
    /// <summary>
    /// List of validation errors
    /// </summary>
    public List<string> Errors { get; } = new List<string>();

    /// <summary>
    /// List of validation warnings
    /// </summary>
    public List<string> Warnings { get; } = new List<string>();

    /// <summary>
    /// Whether the validation passed
    /// </summary>
    public bool IsValid => Errors.Count == 0;

    /// <summary>
    /// Whether there are any warnings
    /// </summary>
    public bool HasWarnings => Warnings.Count > 0;

    /// <summary>
    /// Add an error to the validation result
    /// </summary>
    public void AddError(string error) => Errors.Add(error);

    /// <summary>
    /// Add a warning to the validation result
    /// </summary>
    public void AddWarning(string warning) => Warnings.Add(warning);

    /// <summary>
    /// Add errors to the validation result
    /// </summary>
    public void AddErrors(List<string> errors)
    {
        Errors.AddRange(errors);
    }

    /// <summary>
    /// Get a summary of the validation results
    /// </summary>
    public string GetSummary()
    {
        var summary = new System.Text.StringBuilder();
        summary.AppendLine($"Validation: {(IsValid ? "PASSED" : "FAILED")}");

        if (HasWarnings)
        {
            summary.AppendLine($"Warnings ({Warnings.Count}):");
            foreach (var warning in Warnings)
            {
                summary.AppendLine($"  - {warning}");
            }
        }

        if (Errors.Count > 0)
        {
            summary.AppendLine($"Errors ({Errors.Count}):");
            foreach (var error in Errors)
            {
                summary.AppendLine($"  - {error}");
            }
        }

        return summary.ToString();
    }
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
        var result = new ValidationResult();

        if (metadata == null)
        {
            result.AddError("Metadata cannot be null");
            return result;
        }

        // Validate version
        if (string.IsNullOrWhiteSpace(metadata.Version))
        {
            result.AddError("Version is required");
        }

        // Validate sharding info
        if (metadata.Sharding == null)
        {
            result.AddError("Sharding metadata is required");
        }

        // Validate shards
        if (metadata.Shards == null || metadata.Shards.Count == 0)
        {
            result.AddError("At least one shard is required");
        }
        else if (metadata.Sharding != null && metadata.Shards.Count != metadata.Sharding.ShardCount)
        {
            result.AddError($"Shard count mismatch: expected {metadata.Sharding.ShardCount}, found {metadata.Shards.Count}");
        }
        else
        {
            // Check for duplicate ranks
            var ranks = metadata.Shards.Select(s => s.Rank).ToList();
            var duplicateRanks = ranks.GroupBy(r => r).Where(g => g.Count() > 1).Select(g => g.Key);
            foreach (var rank in duplicateRanks)
            {
                result.AddError($"Duplicate rank found: {rank}");
            }
        }

        // Checksum integrity check (optional - can be deferred)
        if (metadata.Shards != null)
        {
            foreach (var shard in metadata.Shards)
            {
                if (string.IsNullOrEmpty(shard.Checksum))
                {
                    result.AddWarning($"Shard {shard.Rank} missing checksum");
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Validate checkpoint metadata and throw if invalid
    /// </summary>
    public static void ValidateOrThrow(CheckpointMetadata metadata)
    {
        var result = Validate(metadata);
        if (!result.IsValid)
        {
            throw new InvalidOperationException(
                $"Checkpoint metadata validation failed:\n{string.Join("\n", result.Errors)}");
        }
    }
}
