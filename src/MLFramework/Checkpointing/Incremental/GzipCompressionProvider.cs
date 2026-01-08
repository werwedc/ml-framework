namespace MachineLearning.Checkpointing;

using System.IO.Compression;

/// <summary>
/// GZIP compression provider (simplified version using built-in compression)
/// </summary>
public class GzipCompressionProvider : ICompressionProvider
{
    private readonly CompressionLevel _compressionLevel;

    /// <summary>
    /// Create a new GzipCompressionProvider
    /// </summary>
    public GzipCompressionProvider(CompressionLevel compressionLevel = CompressionLevel.Optimal)
    {
        _compressionLevel = compressionLevel;
    }

    /// <inheritdoc/>
    public async Task<byte[]> CompressAsync(byte[] data, CancellationToken cancellationToken = default)
    {
        using var output = new MemoryStream();
        using (var gzip = new GZipStream(output, _compressionLevel))
        {
            await gzip.WriteAsync(data, cancellationToken);
        }

        return output.ToArray();
    }

    /// <inheritdoc/>
    public async Task<byte[]> DecompressAsync(byte[] data, CancellationToken cancellationToken = default)
    {
        using var input = new MemoryStream(data);
        using var gzip = new GZipStream(input, CompressionMode.Decompress);
        using var output = new MemoryStream();

        await gzip.CopyToAsync(output, cancellationToken);
        return output.ToArray();
    }
}
