namespace MachineLearning.Checkpointing;

using System.Security.Cryptography;

/// <summary>
/// Checker for data integrity using checksums
/// </summary>
public class ChecksumIntegrityChecker : IIntegrityChecker
{
    /// <summary>
    /// Name of the integrity checker
    /// </summary>
    public string Name => "Checksum";

    /// <summary>
    /// Check the integrity of data against expected checksum
    /// </summary>
    public async Task<ValidationResult> CheckIntegrityAsync(
        byte[] shardData,
        ShardMetadata shardMeta,
        CancellationToken cancellationToken = default)
    {
        var result = new ValidationResult();

        if (string.IsNullOrEmpty(shardMeta.Checksum))
        {
            result.AddWarning($"Shard {shardMeta.Rank} has no checksum, skipping validation");
            return await Task.FromResult(result);
        }

        var computedChecksum = ComputeChecksum(shardData);
        if (computedChecksum != shardMeta.Checksum)
        {
            result.AddError($"Shard {shardMeta.Rank} checksum mismatch: expected {shardMeta.Checksum}, computed {computedChecksum}");
        }

        return await Task.FromResult(result);
    }

    /// <summary>
    /// Compute SHA-256 checksum of data
    /// </summary>
    private string ComputeChecksum(byte[] data)
    {
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(data);
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
    }
}
