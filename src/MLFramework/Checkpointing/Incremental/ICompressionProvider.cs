namespace MachineLearning.Checkpointing;

/// <summary>
/// Interface for compression providers
/// </summary>
public interface ICompressionProvider
{
    /// <summary>
    /// Compress data
    /// </summary>
    Task<byte[]> CompressAsync(byte[] data, CancellationToken cancellationToken = default);

    /// <summary>
    /// Decompress data
    /// </summary>
    Task<byte[]> DecompressAsync(byte[] data, CancellationToken cancellationToken = default);
}
