namespace MachineLearning.Checkpointing;

/// <summary>
/// Interface for computing checksums of tensor data
/// </summary>
public interface IChecksumCalculator
{
    /// <summary>
    /// Calculate checksum for tensor data
    /// </summary>
    Task<string> CalculateChecksumAsync(float[] data, CancellationToken cancellationToken = default);
}
