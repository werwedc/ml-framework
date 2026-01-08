namespace MachineLearning.Checkpointing;

/// <summary>
/// Local file system storage implementation for checkpoints
/// </summary>
public class LocalFileSystemStorage : ICheckpointStorage
{
    private readonly string _basePath;

    /// <summary>
    /// Creates a new LocalFileSystemStorage instance
    /// </summary>
    /// <param name="basePath">Base directory for storage operations</param>
    public LocalFileSystemStorage(string basePath)
    {
        if (string.IsNullOrWhiteSpace(basePath))
            throw new ArgumentException("Base path cannot be empty", nameof(basePath));

        _basePath = basePath;
        Directory.CreateDirectory(_basePath);
    }

    /// <summary>
    /// Write data to the specified path
    /// </summary>
    public async Task WriteAsync(string path, byte[] data, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path cannot be empty", nameof(path));

        if (data == null)
            throw new ArgumentNullException(nameof(data));

        var fullPath = Path.Combine(_basePath, path);
        var directory = Path.GetDirectoryName(fullPath);

        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        await File.WriteAllBytesAsync(fullPath, data, cancellationToken);
    }

    /// <summary>
    /// Read data from the specified path
    /// </summary>
    public async Task<byte[]> ReadAsync(string path, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path cannot be empty", nameof(path));

        var fullPath = Path.Combine(_basePath, path);

        if (!File.Exists(fullPath))
            throw new FileNotFoundException($"File not found: {fullPath}");

        return await File.ReadAllBytesAsync(fullPath, cancellationToken);
    }

    /// <summary>
    /// Check if a file exists at the specified path
    /// </summary>
    public Task<bool> ExistsAsync(string path, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path cannot be empty", nameof(path));

        var fullPath = Path.Combine(_basePath, path);
        return Task.FromResult(File.Exists(fullPath));
    }

    /// <summary>
    /// Delete a file at the specified path
    /// </summary>
    public async Task DeleteAsync(string path, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path cannot be empty", nameof(path));

        var fullPath = Path.Combine(_basePath, path);

        if (File.Exists(fullPath))
        {
            await Task.Run(() => File.Delete(fullPath), cancellationToken);
        }
    }

    /// <summary>
    /// Get metadata for a file at the specified path
    /// </summary>
    public Task<FileMetadata> GetMetadataAsync(string path, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path cannot be empty", nameof(path));

        var fullPath = Path.Combine(_basePath, path);

        if (!File.Exists(fullPath))
            throw new FileNotFoundException($"File not found: {fullPath}");

        var fileInfo = new FileInfo(fullPath);
        var metadata = new FileMetadata
        {
            Size = fileInfo.Length,
            LastModified = fileInfo.LastWriteTimeUtc
        };

        return Task.FromResult(metadata);
    }
}
