namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Reports capacity utilization statistics.
/// </summary>
/// <param name="SlotUtilization">Percentage of slots used (0-100)</param>
/// <param name="MemoryUtilization">Percentage of memory used (0-100)</param>
/// <param name="ActiveRequestCount">Number of active requests</param>
/// <param name="TotalMemoryUsedBytes">Total memory used in bytes</param>
/// <param name="AvailableMemoryBytes">Available memory in bytes</param>
public record class CapacityUtilization(
    double SlotUtilization,
    double MemoryUtilization,
    int ActiveRequestCount,
    long TotalMemoryUsedBytes,
    long AvailableMemoryBytes
)
{
    /// <summary>
    /// Gets the average utilization of slot and memory.
    /// </summary>
    public double AverageUtilization =>
        (SlotUtilization + MemoryUtilization) / 2.0;
}
