namespace MachineLearning.Checkpointing;

/// <summary>
/// Interface for checkpoint integrity validation
/// </summary>
public interface IIntegrityChecker
{
    /// <summary>
    /// Name of the integrity checker
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Check the integrity of shard data
    /// </summary>
    Task<ValidationResult> CheckIntegrityAsync(
        byte[] shardData,
        ShardMetadata shardMeta,
        CancellationToken cancellationToken = default);
}
