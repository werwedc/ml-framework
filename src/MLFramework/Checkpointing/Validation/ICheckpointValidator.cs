namespace MachineLearning.Checkpointing;

/// <summary>
/// Interface for checkpoint validation
/// </summary>
public interface ICheckpointValidator
{
    /// <summary>
    /// Validate a checkpoint
    /// </summary>
    Task<ValidationResult> ValidateCheckpointAsync(
        string checkpointPath,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Validate checkpoint metadata
    /// </summary>
    Task<ValidationResult> ValidateMetadataAsync(
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Validate checkpoint shards
    /// </summary>
    Task<ValidationResult> ValidateShardsAsync(
        CheckpointMetadata metadata,
        string checkpointPrefix,
        CancellationToken cancellationToken = default);
}
