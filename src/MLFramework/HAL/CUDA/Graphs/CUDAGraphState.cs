namespace MLFramework.HAL.CUDA;

/// <summary>
/// Possible states for a CUDA graph
/// </summary>
public enum CUDAGraphState
{
    /// <summary>
    /// Graph has been created but not yet captured
    /// </summary>
    Created,

    /// <summary>
    /// Graph is currently being captured
    /// </summary>
    Capturing,

    /// <summary>
    /// Graph has been captured and is ready for execution
    /// </summary>
    Ready,

    /// <summary>
    /// Graph is currently executing
    /// </summary>
    Executing,

    /// <summary>
    /// Graph has been invalidated and cannot be executed
    /// </summary>
    Invalidated,

    /// <summary>
    /// Graph has been disposed
    /// </summary>
    Disposed
}
