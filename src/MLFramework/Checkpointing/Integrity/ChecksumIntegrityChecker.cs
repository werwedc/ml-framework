namespace MachineLearning.Checkpointing;

using System.Security.Cryptography;

/// <summary>
/// Result of integrity check
/// </summary>
public class IntegrityCheckResult
{
    /// <summary>
    /// Whether the integrity check passed
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// List of errors
    /// </summary>
    public List<string> Errors { get; set; } = new List<string>();

    /// <summary>
    /// Create a successful integrity check result
    /// </summary>
    public static IntegrityCheckResult Success() => new IntegrityCheckResult { IsValid = true };

    /// <summary>
    /// Create a failed integrity check result with errors
    /// </summary>
    public static IntegrityCheckResult Failure(params string[] errors) =>
        new IntegrityCheckResult { IsValid = false, Errors = errors.ToList() };
}

/// <summary>
/// Checker for data integrity using checksums
/// </summary>
public class ChecksumIntegrityChecker
{
    /// <summary>
    /// Check the integrity of data against expected checksum
    /// </summary>
    public async Task<IntegrityCheckResult> CheckIntegrityAsync(byte[] data, ShardMetadata shardMetadata, CancellationToken cancellationToken = default)
    {
        if (data == null)
            throw new ArgumentNullException(nameof(data));

        if (shardMetadata == null)
            throw new ArgumentNullException(nameof(shardMetadata));

        var errors = new List<string>();

        // Calculate checksum of the data
        var actualChecksum = ComputeChecksum(data);
        var expectedChecksum = shardMetadata.Checksum;

        if (string.IsNullOrWhiteSpace(expectedChecksum))
            errors.Add("Checksum is missing from shard metadata");
        else if (!ChecksumsMatch(actualChecksum, expectedChecksum))
            errors.Add($"Checksum mismatch: expected {expectedChecksum}, got {actualChecksum}");

        return errors.Count == 0 ? IntegrityCheckResult.Success() : IntegrityCheckResult.Failure(errors.ToArray());
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

    /// <summary>
    /// Check if two checksums match (case-insensitive)
    /// </summary>
    private bool ChecksumsMatch(string checksum1, string checksum2)
    {
        return string.Equals(checksum1, checksum2, StringComparison.OrdinalIgnoreCase);
    }
}
