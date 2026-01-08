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
}
