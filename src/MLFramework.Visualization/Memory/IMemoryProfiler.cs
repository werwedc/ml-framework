using System.Diagnostics;

namespace MachineLearning.Visualization.Memory;

/// <summary>
/// Interface for memory profiling functionality
/// </summary>
public interface IMemoryProfiler
{
    // Event tracking

    /// <summary>
    /// Tracks a memory allocation
    /// </summary>
    /// <param name="address">Memory address of the allocation</param>
    /// <param name="sizeBytes">Size of the allocation in bytes</param>
    /// <param name="allocationType">Type of allocation (e.g., "GPU", "CPU", "Pinned")</param>
    void TrackAllocation(long address, long sizeBytes, string allocationType);

    /// <summary>
    /// Tracks a memory deallocation
    /// </summary>
    /// <param name="address">Memory address of the deallocation</param>
    /// <param name="sizeBytes">Size of the deallocation in bytes</param>
    void TrackDeallocation(long address, long sizeBytes);

    /// <summary>
    /// Tracks a snapshot of current memory usage
    /// </summary>
    void TrackSnapshot();

    // Statistics

    /// <summary>
    /// Gets current memory statistics
    /// </summary>
    /// <returns>Current memory statistics</returns>
    MemoryStatistics GetStatistics();

    /// <summary>
    /// Gets memory statistics for a specific allocation type
    /// </summary>
    /// <param name="allocationType">Type of allocation (e.g., "GPU", "CPU", "Pinned")</param>
    /// <returns>Memory statistics for the specified type</returns>
    MemoryStatistics GetStatisticsForType(string allocationType);

    // Timeline

    /// <summary>
    /// Gets memory events within a step range
    /// </summary>
    /// <param name="startStep">Starting step</param>
    /// <param name="endStep">Ending step</param>
    /// <returns>Memory events in the specified step range</returns>
    IEnumerable<MemoryEvent> GetEvents(long startStep, long endStep);

    /// <summary>
    /// Gets allocations since a specific time
    /// </summary>
    /// <param name="startTime">Start time</param>
    /// <returns>Memory events since the specified time</returns>
    IEnumerable<MemoryEvent> GetAllocationsSince(DateTime startTime);

    // Leak detection

    /// <summary>
    /// Detects potential memory leaks by finding allocations without matching deallocations
    /// </summary>
    /// <returns>List of potential leaks with address, size, and stack trace</returns>
    List<(long address, long size, StackTrace? trace)> DetectPotentialLeaks();

    // Configuration

    /// <summary>
    /// Enables memory profiling
    /// </summary>
    void Enable();

    /// <summary>
    /// Disables memory profiling
    /// </summary>
    void Disable();

    /// <summary>
    /// Gets whether memory profiling is enabled
    /// </summary>
    bool IsEnabled { get; }

    /// <summary>
    /// Gets or sets whether to capture stack traces for allocations
    /// </summary>
    bool CaptureStackTraces { get; set; }
}
