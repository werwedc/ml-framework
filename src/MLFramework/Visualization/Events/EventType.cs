namespace MachineLearning.Visualization.Events;

/// <summary>
/// Enumeration of all event types in the visualization system
/// </summary>
public enum EventType
{
    /// <summary>
    /// Scalar metric event (e.g., loss, accuracy)
    /// </summary>
    ScalarMetric,

    /// <summary>
    /// Histogram event (e.g., weight/gradient distributions)
    /// </summary>
    Histogram,

    /// <summary>
    /// Computational graph event (e.g., model architecture)
    /// </summary>
    ComputationalGraph,

    /// <summary>
    /// Profiling start event (beginning of a timed operation)
    /// </summary>
    ProfilingStart,

    /// <summary>
    /// Profiling end event (end of a timed operation)
    /// </summary>
    ProfilingEnd,

    /// <summary>
    /// Memory allocation event (e.g., tensor allocation)
    /// </summary>
    MemoryAllocation,

    /// <summary>
    /// Tensor operation event (e.g., matrix multiplication)
    /// </summary>
    TensorOperation,

    /// <summary>
    /// Custom event type for user-defined events
    /// </summary>
    Custom
}
