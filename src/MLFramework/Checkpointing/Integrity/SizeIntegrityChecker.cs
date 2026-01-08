namespace MachineLearning.Checkpointing;

/// <summary>
/// Checker for data integrity using file size validation
/// </summary>
public class SizeIntegrityChecker : IIntegrityChecker
{
    /// <summary>
    /// Name of the integrity checker
    /// </summary>
    public string Name => "Size";

    /// <summary>
    /// Check the integrity of data against expected file size
    /// </summary>
    public async Task<ValidationResult> CheckIntegrityAsync(
        byte[] shardData,
        ShardMetadata shardMeta,
        CancellationToken cancellationToken = default)
    {
        var result = new ValidationResult();

        if (shardData.Length != shardMeta.FileSize)
        {
            result.AddError($"Shard {shardMeta.Rank} size mismatch: expected {shardMeta.FileSize}, found {shardData.Length}");
        }

        return await Task.FromResult(result);
    }
}
