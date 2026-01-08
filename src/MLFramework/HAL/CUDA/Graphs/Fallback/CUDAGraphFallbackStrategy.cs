namespace MLFramework.HAL.CUDA.Graphs;

/// <summary>
/// Defines different fallback strategies for CUDA graph capture
/// </summary>
public enum CUDAGraphFallbackStrategy
{
    /// <summary>
    /// Always use regular execution, never capture
    /// </summary>
    NeverCapture,

    /// <summary>
    /// Capture if possible, fallback to regular execution on failure
    /// </summary>
    CaptureOrFallback,

    /// <summary>
    /// Capture only, throw exception on failure
    /// </summary>
    CaptureOnly,

    /// <summary>
    /// Try capture once, then permanently fallback on failure
    /// </summary>
    TryOnceThenFallback,

    /// <summary>
    /// Try capture multiple times before falling back
    /// </summary>
    RetryThenFallback
}
