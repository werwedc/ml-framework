using Microsoft.Extensions.Logging;

namespace MachineLearning.Checkpointing.FaultTolerance;

/// <summary>
/// Manager for checkpoint rollback operations
/// </summary>
public class CheckpointRollbackManager
{
    private readonly IFaultToleranceHandler _faultHandler;
    private readonly ICheckpointStorage _storage;
    private readonly ILogger<CheckpointRollbackManager>? _logger;

    /// <summary>
    /// Create a new CheckpointRollbackManager
    /// </summary>
    public CheckpointRollbackManager(
        IFaultToleranceHandler faultHandler,
        ICheckpointStorage storage,
        ILogger<CheckpointRollbackManager>? logger = null)
    {
        _faultHandler = faultHandler ?? throw new ArgumentNullException(nameof(faultHandler));
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
        _logger = logger;
    }

    /// <summary>
    /// Execute checkpoint save with automatic rollback on failure
    /// </summary>
    public async Task<string> SaveWithRollbackAsync(
        string checkpointPath,
        Func<Task<byte[]>> saveOperation,
        CancellationToken cancellationToken = default)
    {
        byte[]? tempData = null;
        string? tempPath = null;

        try
        {
            // Save to temporary location first
            tempData = await _faultHandler.ExecuteWithRetryAsync(saveOperation, cancellationToken);
            tempPath = $"{checkpointPath}.tmp";
            await _storage.WriteAsync(tempPath, tempData, cancellationToken);

            // Atomic move to final location
            if (_storage is LocalFileSystemStorage localStorage)
            {
                File.Move(tempPath, checkpointPath, overwrite: true);
            }
            else
            {
                // For cloud storage, write to final location
                await _storage.WriteAsync(checkpointPath, tempData, cancellationToken);
                await _storage.DeleteAsync(tempPath, cancellationToken);
            }

            return checkpointPath;
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Checkpoint save failed, rolling back: {CheckpointPath}", checkpointPath);

            // Cleanup temporary file
            if (tempPath != null)
            {
                try
                {
                    await _storage.DeleteAsync(tempPath, cancellationToken);
                }
                catch (Exception cleanupEx)
                {
                    _logger?.LogWarning(cleanupEx, "Failed to cleanup temporary file: {TempPath}", tempPath);
                }
            }

            throw;
        }
    }
}
