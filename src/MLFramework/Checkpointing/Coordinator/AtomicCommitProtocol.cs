namespace MachineLearning.Checkpointing;

/// <summary>
/// Protocol for atomic commit operations in distributed checkpointing
/// </summary>
public class AtomicCommitProtocol
{
    private readonly ICheckpointStorage _storage;

    /// <summary>
    /// Create a new AtomicCommitProtocol instance
    /// </summary>
    /// <param name="storage">Checkpoint storage implementation</param>
    public AtomicCommitProtocol(ICheckpointStorage storage)
    {
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
    }

    /// <summary>
    /// Implement two-phase commit: write to temp, then rename
    /// </summary>
    /// <param name="finalPath">Final path where the file should be stored</param>
    /// <param name="data">Data to write</param>
    /// <param name="cancellationToken">Cancellation token</param>
    public async Task CommitAsync(
        string finalPath,
        byte[] data,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(finalPath))
            throw new ArgumentException("Final path cannot be empty", nameof(finalPath));

        if (data == null)
            throw new ArgumentNullException(nameof(data));

        var tempPath = $"{finalPath}.tmp";

        try
        {
            // Write to temporary file
            await _storage.WriteAsync(tempPath, data, cancellationToken);

            // Atomically move to final location
            if (_storage is LocalFileSystemStorage localStorage)
            {
                await CommitToLocalFileSystemAsync(localStorage, tempPath, finalPath, cancellationToken);
            }
            else
            {
                // For cloud storage, copy and delete
                await _storage.WriteAsync(finalPath, data, cancellationToken);
                await _storage.DeleteAsync(tempPath, cancellationToken);
            }
        }
        catch
        {
            // Cleanup on failure
            try
            {
                await _storage.DeleteAsync(tempPath, cancellationToken);
            }
            catch
            {
                // Ignore cleanup errors
            }
            throw;
        }
    }

    /// <summary>
    /// Commit to local file system using atomic rename
    /// </summary>
    private async Task CommitToLocalFileSystemAsync(
        LocalFileSystemStorage localStorage,
        string tempPath,
        string finalPath,
        CancellationToken cancellationToken)
    {
        // For local file system, we need to handle the path resolution
        // since LocalFileSystemStorage uses a base path
        var basePath = GetBasePath(localStorage);
        var fullTempPath = Path.Combine(basePath, tempPath);
        var fullFinalPath = Path.Combine(basePath, finalPath);

        // Ensure destination directory exists
        var finalDirectory = Path.GetDirectoryName(fullFinalPath);
        if (!string.IsNullOrEmpty(finalDirectory) && !Directory.Exists(finalDirectory))
        {
            Directory.CreateDirectory(finalDirectory);
        }

        // Atomically move the file
        await Task.Run(() =>
        {
            File.Move(
                fullTempPath,
                fullFinalPath,
                overwrite: true);
        }, cancellationToken);
    }

    /// <summary>
    /// Get base path from LocalFileSystemStorage instance
    /// </summary>
    private string GetBasePath(LocalFileSystemStorage storage)
    {
        // Use reflection to get the private _basePath field
        var basePathField = typeof(LocalFileSystemStorage)
            .GetField("_basePath", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

        if (basePathField != null)
        {
            return basePathField.GetValue(storage)?.ToString() ?? string.Empty;
        }

        // Fallback: try to get it through a public property or method if available
        throw new InvalidOperationException("Could not access base path from LocalFileSystemStorage");
    }
}
