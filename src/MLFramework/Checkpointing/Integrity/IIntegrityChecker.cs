namespace MachineLearning.Checkpointing;

/// <summary>
/// Interface for checkpoint integrity validation
/// </summary>
public interface IIntegrityChecker
{
    /// <summary>
    /// Check the integrity of checkpoint data
    /// </summary>
    Task<IntegrityCheckResult> CheckIntegrityAsync(
        byte[] checkpointData,
        CheckpointMetadata? metadata = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Check the integrity of a checkpoint file
    /// </summary>
    Task<IntegrityCheckResult> CheckFileIntegrityAsync(
        string filePath,
        CancellationToken cancellationToken = default);
}
