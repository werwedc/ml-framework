namespace MachineLearning.Checkpointing;

using System.Security.Cryptography;
using System.Text;

/// <summary>
/// SHA-256 checksum calculator for tensor data
/// </summary>
public class SHA256ChecksumCalculator : IChecksumCalculator
{
    /// <inheritdoc/>
    public async Task<string> CalculateChecksumAsync(float[] data, CancellationToken cancellationToken = default)
    {
        using var sha256 = SHA256.Create();

        // Convert float array to bytes
        var dataBytes = new byte[data.Length * sizeof(float)];
        Buffer.BlockCopy(data, 0, dataBytes, 0, dataBytes.Length);

        var hashBytes = await Task.Run(() => sha256.ComputeHash(dataBytes), cancellationToken);
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
    }
}
