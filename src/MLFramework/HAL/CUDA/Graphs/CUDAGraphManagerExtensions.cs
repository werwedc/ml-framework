using System;
using MLFramework.HAL.CUDA;
using MLFramework.HAL.CUDA.Graphs;

namespace MLFramework.HAL.CUDA.Graphs;

/// <summary>
/// Extension methods for common graph management patterns.
/// </summary>
public static class CUDAGraphManagerExtensions
{
    /// <summary>
    /// Gets or captures a graph for a specific phase.
    /// </summary>
    /// <param name="manager">The graph manager</param>
    /// <param name="phase">The training phase</param>
    /// <param name="captureAction">Action to capture for the graph</param>
    /// <param name="stream">CUDA stream for capture/execution</param>
    /// <returns>The captured graph, or null if still in warm-up phase</returns>
    public static ICUDAGraph GetOrCapturePhaseGraph(
        this CUDAGraphManager manager,
        GraphPhase phase,
        Action<CudaStream> captureAction,
        CudaStream stream)
    {
        if (manager == null)
            throw new ArgumentNullException(nameof(manager));

        return manager.GetOrCaptureGraph($"Phase_{phase}", captureAction, stream);
    }

    /// <summary>
    /// Executes a graph for a specific phase.
    /// </summary>
    /// <param name="manager">The graph manager</param>
    /// <param name="phase">The training phase</param>
    /// <param name="captureAction">Action to execute if graph is not ready</param>
    /// <param name="stream">CUDA stream for execution</param>
    public static void ExecutePhaseGraph(
        this CUDAGraphManager manager,
        GraphPhase phase,
        Action<CudaStream> captureAction,
        CudaStream stream)
    {
        if (manager == null)
            throw new ArgumentNullException(nameof(manager));

        manager.ExecuteGraphOrFallback($"Phase_{phase}", captureAction, stream);
    }
}
