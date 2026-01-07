namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Abstraction for GPU resource monitoring.
/// </summary>
public interface IGPUResourceManager
{
    /// <summary>
    /// Gets the total GPU memory available.
    /// </summary>
    /// <returns>Total memory in bytes</returns>
    long GetTotalMemoryBytes();

    /// <summary>
    /// Gets the current GPU memory usage.
    /// </summary>
    /// <returns>Current memory usage in bytes</returns>
    long GetCurrentMemoryUsageBytes();

    /// <summary>
    /// Gets the GPU utilization percentage (0-100).
    /// </summary>
    /// <returns>Utilization percentage</returns>
    double GetUtilization();

    /// <summary>
    /// Checks if the GPU is available.
    /// </summary>
    /// <returns>True if available, false otherwise</returns>
    bool IsAvailable();
}
