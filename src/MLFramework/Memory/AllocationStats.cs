namespace MLFramework.Memory;

/// <summary>
/// Statistics for memory allocation operations.
/// </summary>
public class AllocationStats
{
    /// <summary>
    /// Total number of allocations performed.
    /// </summary>
    public int TotalAllocations { get; set; }

    /// <summary>
    /// Total number of resize operations.
    /// </summary>
    public int TotalResizes { get; set; }

    /// <summary>
    /// Total bytes allocated across all operations.
    /// </summary>
    public long TotalBytesAllocated { get; set; }

    /// <summary>
    /// Total bytes wasted due to padding overhead.
    /// </summary>
    public long TotalBytesWasted { get; set; }

    /// <summary>
    /// Average utilization ratio (current size / capacity).
    /// </summary>
    public double AverageUtilization { get; set; }

    /// <summary>
    /// Generates a human-readable report of the statistics.
    /// </summary>
    public string ToReport()
    {
        var wastePercentage = TotalBytesAllocated > 0
            ? (TotalBytesWasted * 100.0 / TotalBytesAllocated).ToString("F2")
            : "0.00";

        return $"""
            === Memory Allocation Statistics ===
            Total Allocations:     {TotalAllocations}
            Total Resizes:         {TotalResizes}
            Total Bytes Allocated: {TotalBytesAllocated:N0}
            Total Bytes Wasted:    {TotalBytesWasted:N0} ({wastePercentage}%)
            Average Utilization:   {AverageUtilization:P2}
            ===================================
            """;
    }
}
