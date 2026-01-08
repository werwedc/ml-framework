namespace MachineLearning.Checkpointing;

/// <summary>
/// Checker for schema compatibility validation
/// </summary>
public class SchemaCompatibilityChecker : ICompatibilityChecker
{
    /// <summary>
    /// Name of the compatibility checker
    /// </summary>
    public string Name => "Schema";

    /// <summary>
    /// Check if a checkpoint's schema is compatible with the current setup
    /// </summary>
    public async Task<ValidationResult> CheckCompatibilityAsync(
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default)
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
