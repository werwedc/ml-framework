using System;

namespace MLFramework.HAL.CUDA.Graphs;

/// <summary>
/// Extension methods for CUDA graph fallback functionality
/// </summary>
public static class CUDAGraphFallbackExtensions
{
    /// <summary>
    /// Creates a fallback handler with the specified strategy
    /// </summary>
    /// <param name="manager">The CUDA graph manager</param>
    /// <param name="strategy">Fallback strategy to use</param>
    /// <param name="maxRetries">Maximum number of retries for RetryThenFallback strategy</param>
    /// <returns>A new fallback handler instance</returns>
    public static CUDAGraphFallbackHandler WithFallback(
        this CUDAGraphManager manager,
        CUDAGraphFallbackStrategy strategy = CUDAGraphFallbackStrategy.CaptureOrFallback,
        int maxRetries = 3)
    {
        return new CUDAGraphFallbackHandler(strategy, maxRetries);
    }

    /// <summary>
    /// Executes with automatic fallback using the graph manager
    /// </summary>
    /// <param name="handler">The fallback handler</param>
    /// <param name="graphName">Name of the graph to capture/execute</param>
    /// <param name="captureAction">Action to capture for the graph</param>
    /// <param name="stream">CUDA stream for execution</param>
    /// <param name="manager">The CUDA graph manager</param>
    public static void ExecuteWithFallback(
        this CUDAGraphFallbackHandler handler,
        string graphName,
        Action<CudaStream> captureAction,
        CudaStream stream,
        CUDAGraphManager manager)
    {
        handler.TryExecuteWithFallback(
            () => manager.GetOrCaptureGraph(graphName, captureAction, stream)!,
            captureAction,
            stream,
            out _);
    }
}
