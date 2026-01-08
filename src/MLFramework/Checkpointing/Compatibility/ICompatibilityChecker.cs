namespace MachineLearning.Checkpointing;

/// <summary>
/// Interface for checkpoint compatibility validation
/// </summary>
public interface ICompatibilityChecker
{
    /// <summary>
    /// Name of the compatibility checker
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Check if a checkpoint is compatible with the current setup
    /// </summary>
    Task<ValidationResult> CheckCompatibilityAsync(
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default);
}
