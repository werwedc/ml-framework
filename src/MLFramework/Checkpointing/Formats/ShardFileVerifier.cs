namespace MachineLearning.Checkpointing;

using System.Security.Cryptography;

/// <summary>
/// Verifies integrity of multi-shard checkpoint files
/// </summary>
public class ShardFileVerifier
{
    private readonly ICheckpointStorage _storage;

    /// <summary>
    /// Create a new shard file verifier
    /// </summary>
    /// <param name="storage">Checkpoint storage implementation</param>
    public ShardFileVerifier(ICheckpointStorage storage)
    {
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
    }

    /// <summary>
    /// Verify all shard files exist and match metadata
    /// </summary>
    /// <param name="checkpointPrefix">Checkpoint prefix</param>
    /// <param name="metadata">Checkpoint metadata to verify against</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Verification result with errors and warnings</returns>
    public async Task<VerificationResult> VerifyCheckpointAsync(
        string checkpointPrefix,
        CheckpointMetadata metadata,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(checkpointPrefix))
            throw new ArgumentException("Checkpoint prefix cannot be empty", nameof(checkpointPrefix));

        if (metadata == null)
            throw new ArgumentNullException(nameof(metadata));

        cancellationToken.ThrowIfCancellationRequested();

        var result = new VerificationResult();

        // Verify each shard
        if (metadata.Shards != null)
        {
            foreach (var shardMeta in metadata.Shards)
            {
                await VerifyShardAsync(checkpointPrefix, shardMeta, result, cancellationToken);
            }
        }

        // Verify metadata file
        var metadataPath = $"{checkpointPrefix}.metadata.json";
        var metadataExists = await _storage.ExistsAsync(metadataPath, cancellationToken);
        if (!metadataExists)
        {
            result.AddError($"Metadata file not found: {metadataPath}");
        }

        return result;
    }

    /// <summary>
    /// Verify a single shard file
    /// </summary>
    /// <param name="checkpointPrefix">Checkpoint prefix</param>
    /// <param name="shardMeta">Shard metadata</param>
    /// <param name="result">Verification result to update</param>
    /// <param name="cancellationToken">Cancellation token</param>
    private async Task VerifyShardAsync(
        string checkpointPrefix,
        ShardMetadata shardMeta,
        VerificationResult result,
        CancellationToken cancellationToken)
    {
        var shardPath = $"{checkpointPrefix}_shard_{shardMeta.Rank}.shard";

        // Check existence
        var exists = await _storage.ExistsAsync(shardPath, cancellationToken);
        if (!exists)
        {
            result.AddError($"Shard file not found: {shardPath}");
            return;
        }

        // Check file size
        try
        {
            var fileMetadata = await _storage.GetMetadataAsync(shardPath, cancellationToken);
            if (fileMetadata.Size != shardMeta.FileSize)
            {
                result.AddError(
                    $"Shard file size mismatch: {shardPath} " +
                    $"(expected {shardMeta.FileSize}, found {fileMetadata.Size})");
            }
        }
        catch (Exception ex)
        {
            result.AddError($"Failed to get metadata for shard {shardPath}: {ex.Message}");
        }

        // Verify checksum (optional - can be deferred)
        if (!string.IsNullOrEmpty(shardMeta.Checksum))
        {
            try
            {
                var data = await _storage.ReadAsync(shardPath, cancellationToken);
                var computedChecksum = ComputeChecksum(data);
                if (computedChecksum != shardMeta.Checksum)
                {
                    result.AddError($"Shard checksum mismatch: {shardPath}");
                }
            }
            catch (Exception ex)
            {
                result.AddError($"Failed to verify checksum for shard {shardPath}: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Compute SHA-256 checksum of data
    /// </summary>
    /// <param name="data">Data to compute checksum for</param>
    /// <returns>Hex-encoded checksum string</returns>
    private string ComputeChecksum(byte[] data)
    {
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(data);
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
    }
}
