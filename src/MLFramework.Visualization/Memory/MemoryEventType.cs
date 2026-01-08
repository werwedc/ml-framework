namespace MachineLearning.Visualization.Memory;

/// <summary>
/// Types of memory events that can be tracked
/// </summary>
public enum MemoryEventType
{
    /// <summary>
    /// Memory was allocated
    /// </summary>
    Allocation,

    /// <summary>
    /// Memory was freed
    /// </summary>
    Deallocation,

    /// <summary>
    /// Memory was resized/reallocated
    /// </summary>
    Reallocation,

    /// <summary>
    /// A snapshot of current memory usage
    /// </summary>
    Snapshot,

    /// <summary>
    /// Garbage collection started
    /// </summary>
    GCStart,

    /// <summary>
    /// Garbage collection ended
    /// </summary>
    GCEnd
}
