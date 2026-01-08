namespace MachineLearning.Checkpointing;

/// <summary>
/// Interface for checkpoint storage implementations
/// </summary>
public interface ICheckpointStorage
{
    /// <summary>
    /// Write data to the specified path
    /// </summary>
    Task WriteAsync(string path, byte[] data, CancellationToken cancellationToken = default);

    /// <summary>
    /// Read data from the specified path
    /// </summary>
    Task<byte[]> ReadAsync(string path, CancellationToken cancellationToken = default);

    /// <summary>
    /// Check if a file exists at the specified path
    /// </summary>
    Task<bool> ExistsAsync(string path, CancellationToken cancellationToken = default);

    /// <summary>
    /// Delete a file at the specified path
    /// </summary>
    Task DeleteAsync(string path, CancellationToken cancellationToken = default);

    /// <summary>
    /// Get metadata for a file at the specified path
    /// </summary>
    Task<FileMetadata> GetMetadataAsync(string path, CancellationToken cancellationToken = default);
}

/// <summary>
/// File metadata
/// </summary>
public class FileMetadata
{
    /// <summary>
    /// Size of the file in bytes
    /// </summary>
    public long Size { get; set; }

    /// <summary>
    /// Last modified timestamp
    /// </summary>
    public DateTime LastModified { get; set; }
}
