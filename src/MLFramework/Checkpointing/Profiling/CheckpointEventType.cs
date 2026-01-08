namespace MLFramework.Checkpointing.Profiling;

/// <summary>
/// Types of checkpoint events
/// </summary>
public enum CheckpointEventType
{
    /// <summary>
    /// Checkpoint registration
    /// </summary>
    Checkpoint,

    /// <summary>
    /// Activation recomputation
    /// </summary>
    Recompute,

    /// <summary>
    /// Checkpoint retrieval (cache hit)
    /// </summary>
    Retrieve,

    /// <summary>
    /// Checkpoint deallocation
    /// </summary>
    Deallocate
}
