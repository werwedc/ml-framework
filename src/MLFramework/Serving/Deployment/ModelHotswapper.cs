using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;

namespace MLFramework.Serving.Deployment;

/// <summary>
/// Implementation of model hot-swapping for zero-downtime version updates
/// </summary>
public class ModelHotswapper : IModelHotswapper
{
    private readonly IModelLoader _modelLoader;
    private readonly IVersionRouterCore _versionRouterCore;
    private readonly IModelRegistry _modelRegistry;
    private readonly ConcurrentDictionary<string, SwapOperation> _activeSwaps;
    private readonly ConcurrentDictionary<string, ConcurrentBag<string>> _modelSwapQueue;

    /// <summary>
    /// Create a new ModelHotswapper instance
    /// </summary>
    public ModelHotswapper(
        IModelLoader modelLoader,
        IVersionRouterCore versionRouterCore,
        IModelRegistry modelRegistry)
    {
        _modelLoader = modelLoader ?? throw new ArgumentNullException(nameof(modelLoader));
        _versionRouterCore = versionRouterCore ?? throw new ArgumentNullException(nameof(versionRouterCore));
        _modelRegistry = modelRegistry ?? throw new ArgumentNullException(nameof(modelRegistry));
        _activeSwaps = new ConcurrentDictionary<string, SwapOperation>();
        _modelSwapQueue = new ConcurrentDictionary<string, ConcurrentBag<string>>();
    }

    /// <summary>
    /// Swap from one version to another without dropping requests
    /// </summary>
    public async Task<SwapOperation> SwapVersionAsync(string modelName, string fromVersion, string toVersion)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));
        if (string.IsNullOrWhiteSpace(fromVersion))
            throw new ArgumentException("From version cannot be null or empty", nameof(fromVersion));
        if (string.IsNullOrWhiteSpace(toVersion))
            throw new ArgumentException("To version cannot be null or empty", nameof(toVersion));

        // Check if there's already an active swap for this model
        if (IsHotswapInProgress(modelName))
            throw new InvalidOperationException($"A swap is already in progress for model '{modelName}'");

        // Validate fromVersion is active and toVersion exists
        if (!IsVersionActive(modelName, fromVersion))
            throw new InvalidOperationException($"Source version '{fromVersion}' is not active for model '{modelName}'");

        if (!_modelRegistry.HasVersion(modelName, toVersion))
            throw new InvalidOperationException($"Target version '{toVersion}' does not exist for model '{modelName}'");

        // Create and register the swap operation
        var operationId = Guid.NewGuid().ToString();
        var operation = new SwapOperation(operationId, modelName, fromVersion, toVersion);
        _activeSwaps.TryAdd(operationId, operation);

        try
        {
            // Step 1: Load toVersion (async, in background)
            operation.State = SwapState.LoadingNewVersion;
            LogSwapProgress(operation, "Loading new version");

            // Check if target version is already loaded
            if (!_modelLoader.IsLoaded(modelName, toVersion))
            {
                // This is a simplified version - in a real implementation,
                // we would load the model from its path
                // For now, we assume the model is already registered in the registry
                LogSwapProgress(operation, "New version verified in registry");
            }

            // Step 2: Mark swap as "transitioning"
            operation.State = SwapState.Transitioning;
            LogSwapProgress(operation, "Transitioning to new version");

            // Step 3: Update router to route new requests to toVersion
            await _versionRouterCore.UpdateRoutingAsync(modelName, toVersion);
            LogSwapProgress(operation, "Routing updated to new version");

            // Step 4: Wait for old version to drain (no more active requests)
            operation.State = SwapState.OldVersionDraining;
            LogSwapProgress(operation, "Waiting for old version to drain");

            // Wait for the old version to drain (default 30s timeout)
            WaitForDrainage(modelName, fromVersion, TimeSpan.FromSeconds(30));

            // Step 5: Unload fromVersion
            LogSwapProgress(operation, "Unloading old version");
            // Note: In a real implementation, we would call _modelLoader.Unload here
            // For now, we'll just mark it as inactive

            // Step 6: Mark swap as "completed"
            operation.State = SwapState.Completed;
            operation.EndTime = DateTime.UtcNow;
            LogSwapProgress(operation, "Swap completed successfully");

            return operation;
        }
        catch (Exception ex)
        {
            operation.State = SwapState.Failed;
            operation.EndTime = DateTime.UtcNow;
            operation.ErrorMessage = ex.Message;
            LogSwapProgress(operation, $"Swap failed: {ex.Message}");
            throw;
        }
        finally
        {
            // Remove from active swaps after a delay
            _ = Task.Run(async () =>
            {
                await Task.Delay(TimeSpan.FromMinutes(1));
                _activeSwaps.TryRemove(operationId, out _);
            });
        }
    }

    /// <summary>
    /// Get the status of a swap operation
    /// </summary>
    public SwapOperation GetSwapStatus(string operationId)
    {
        if (string.IsNullOrWhiteSpace(operationId))
            throw new ArgumentException("Operation ID cannot be null or empty", nameof(operationId));

        if (_activeSwaps.TryGetValue(operationId, out var operation))
            return operation;

        throw new KeyNotFoundException($"Swap operation '{operationId}' not found");
    }

    /// <summary>
    /// Wait for the current version to drain (complete in-flight requests)
    /// </summary>
    public void WaitForDrainage(string modelName, string version, TimeSpan timeout)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));
        if (string.IsNullOrWhiteSpace(version))
            throw new ArgumentException("Version cannot be null or empty", nameof(version));

        var startTime = DateTime.UtcNow;
        var drained = false;

        while (DateTime.UtcNow - startTime < timeout)
        {
            // Check if there are any active requests to this version
            // In a real implementation, this would check actual request counters
            // For now, we assume the drain is complete if there are no pending operations

            // Simulate draining by checking if there are any active swaps
            var hasActiveSwaps = _activeSwaps.Values
                .Any(op => op.ModelName == modelName && op.State == SwapState.OldVersionDraining);

            if (!hasActiveSwaps)
            {
                drained = true;
                break;
            }

            Thread.Sleep(100);
        }

        if (!drained)
            throw new TimeoutException($"Timed out waiting for version '{version}' of model '{modelName}' to drain");
    }

    /// <summary>
    /// Check if a version is currently active
    /// </summary>
    public bool IsVersionActive(string modelName, string version)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Model name cannot be null or empty", nameof(modelName));
        if (string.IsNullOrWhiteSpace(version))
            throw new ArgumentException("Version cannot be null or empty", nameof(version));

        // Check if the version is loaded in the loader
        return _modelLoader.IsLoaded(modelName, version);
    }

    /// <summary>
    /// Rollback to the previous version
    /// </summary>
    public async Task RollbackAsync(string operationId)
    {
        if (string.IsNullOrWhiteSpace(operationId))
            throw new ArgumentException("Operation ID cannot be null or empty", nameof(operationId));

        if (!_activeSwaps.TryGetValue(operationId, out var operation))
            throw new KeyNotFoundException($"Swap operation '{operationId}' not found");

        if (operation.State == SwapState.Completed)
            throw new InvalidOperationException($"Cannot rollback a completed swap operation");

        if (operation.State == SwapState.Failed)
            throw new InvalidOperationException($"Cannot rollback a failed swap operation");

        try
        {
            LogSwapProgress(operation, "Starting rollback");

            // Update routing back to the original version
            await _versionRouterCore.UpdateRoutingAsync(operation.ModelName, operation.FromVersion);

            LogSwapProgress(operation, "Rollback completed successfully");
        }
        catch (Exception ex)
        {
            operation.State = SwapState.Failed;
            operation.EndTime = DateTime.UtcNow;
            operation.ErrorMessage = $"Rollback failed: {ex.Message}";
            LogSwapProgress(operation, $"Rollback failed: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// Check if a hotswap is currently in progress
    /// </summary>
    private bool IsHotswapInProgress(string modelName)
    {
        return _activeSwaps.Values
            .Any(op => op.ModelName == modelName &&
                      op.State != SwapState.Completed &&
                      op.State != SwapState.Failed);
    }

    /// <summary>
    /// Log swap progress
    /// </summary>
    private void LogSwapProgress(SwapOperation operation, string message)
    {
        // In a real implementation, this would use a proper logging framework
        // For now, we'll use console output for debugging
        var timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss.fff");
        Console.WriteLine($"[{timestamp}] Swap {operation.OperationId} ({operation.State}): {message}");
    }
}
