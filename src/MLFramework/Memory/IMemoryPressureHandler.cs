using Microsoft.Extensions.Logging;

namespace MLFramework.Memory;

/// <summary>
/// Interface for handling memory pressure and evicting least-recently-used models.
/// </summary>
public interface IMemoryPressureHandler
{
    /// <summary>
    /// Tracks a newly loaded model in memory.
    /// </summary>
    /// <param name="modelName">Name of the model.</param>
    /// <param name="version">Version of the model.</param>
    /// <param name="memoryBytes">Memory size in bytes.</param>
    void TrackModelLoad(string modelName, string version, long memoryBytes);

    /// <summary>
    /// Tracks an access to a loaded model.
    /// </summary>
    /// <param name="modelName">Name of the model.</param>
    /// <param name="version">Version of the model.</param>
    void TrackModelAccess(string modelName, string version);

    /// <summary>
    /// Pins a model to prevent it from being evicted.
    /// </summary>
    /// <param name="modelName">Name of the model.</param>
    /// <param name="version">Version of the model.</param>
    void PinModel(string modelName, string version);

    /// <summary>
    /// Unpins a model, allowing it to be evicted again.
    /// </summary>
    /// <param name="modelName">Name of the model.</param>
    /// <param name="version">Version of the model.</param>
    void UnpinModel(string modelName, string version);

    /// <summary>
    /// Gets the total memory usage of all tracked models.
    /// </summary>
    /// <returns>Total memory usage in bytes.</returns>
    long GetTotalMemoryUsage();

    /// <summary>
    /// Sets the memory threshold that triggers eviction.
    /// </summary>
    /// <param name="thresholdBytes">Threshold in bytes.</param>
    void SetMemoryThreshold(long thresholdBytes);

    /// <summary>
    /// Evicts models if memory usage is over threshold or if more space is required.
    /// </summary>
    /// <param name="requiredBytes">Additional bytes required (default 0).</param>
    /// <returns>Eviction result with details of what was evicted.</returns>
    Task<EvictionResult> EvictIfNeededAsync(long requiredBytes = 0);

    /// <summary>
    /// Gets information about all loaded models.
    /// </summary>
    /// <returns>Collection of model memory info.</returns>
    IEnumerable<ModelMemoryInfo> GetLoadedModelsInfo();

    /// <summary>
    /// Stops tracking a model (e.g., when it's explicitly unloaded).
    /// </summary>
    /// <param name="modelName">Name of the model.</param>
    /// <param name="version">Version of the model.</param>
    void UntrackModel(string modelName, string version);
}
