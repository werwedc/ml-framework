namespace MLFramework.HAL.CUDA;

/// <summary>
/// Static API for explicit CUDA graph capture operations
/// </summary>
public static class CUDAGraphStatic
{
    /// <summary>
    /// Captures a CUDA graph from the specified action
    /// </summary>
    /// <param name="captureAction">Action to capture</param>
    /// <param name="stream">CUDA stream for capture</param>
    /// <returns>Captured graph</returns>
    public static ICUDAGraph Capture(Action<CudaStream> captureAction, CudaStream stream)
    {
        if (captureAction == null)
            throw new ArgumentNullException(nameof(captureAction));

        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        using var capture = new CUDAGraphCapture();
        capture.BeginCapture(stream);

        captureAction(stream);

        return capture.EndCapture();
    }

    /// <summary>
    /// Captures a CUDA graph with a warm-up phase
    /// </summary>
    /// <param name="captureAction">Action to capture</param>
    /// <param name="stream">CUDA stream for capture</param>
    /// <param name="warmupIterations">Number of warm-up iterations</param>
    /// <returns>Captured graph</returns>
    public static ICUDAGraph CaptureWithWarmup(
        Action<CudaStream> captureAction,
        CudaStream stream,
        int warmupIterations = 3)
    {
        if (captureAction == null)
            throw new ArgumentNullException(nameof(captureAction));

        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (warmupIterations < 0)
            throw new ArgumentOutOfRangeException(nameof(warmupIterations), "Warmup iterations must be non-negative");

        // Warm-up iterations
        for (int i = 0; i < warmupIterations; i++)
        {
            captureAction(stream);
        }

        // Capture
        return Capture(captureAction, stream);
    }

    /// <summary>
    /// Captures a CUDA graph with options
    /// </summary>
    /// <param name="captureAction">Action to capture</param>
    /// <param name="stream">CUDA stream for capture</param>
    /// <param name="options">Capture options</param>
    /// <returns>Captured graph</returns>
    public static ICUDAGraph Capture(
        Action<CudaStream> captureAction,
        CudaStream stream,
        CUDAGraphCaptureOptions options)
    {
        if (captureAction == null)
            throw new ArgumentNullException(nameof(captureAction));

        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        if (options == null)
            throw new ArgumentNullException(nameof(options));

        // Apply warm-up if specified
        if (options.WarmupIterations > 0)
        {
            for (int i = 0; i < options.WarmupIterations; i++)
            {
                captureAction(stream);
            }
        }

        // Capture
        var graph = Capture(captureAction, stream);

        // Validate if requested
        if (options.ValidateOnCapture)
        {
            var validation = graph.Validate();
            if (!validation.IsValid)
            {
                throw new InvalidOperationException(
                    $"Graph validation failed: {string.Join(", ", validation.Errors)}");
            }
        }

        return graph;
    }

    /// <summary>
    /// Captures and immediately executes a graph
    /// </summary>
    /// <param name="captureAction">Action to capture</param>
    /// <param name="stream">CUDA stream for capture and execution</param>
    /// <returns>Captured graph</returns>
    public static ICUDAGraph CaptureAndExecute(
        Action<CudaStream> captureAction,
        CudaStream stream)
    {
        if (captureAction == null)
            throw new ArgumentNullException(nameof(captureAction));

        if (stream == null)
            throw new ArgumentNullException(nameof(stream));

        var graph = Capture(captureAction, stream);
        graph.Execute(stream);
        return graph;
    }
}
