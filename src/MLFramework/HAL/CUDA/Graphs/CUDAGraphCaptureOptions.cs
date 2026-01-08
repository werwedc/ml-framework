namespace MLFramework.HAL.CUDA;

/// <summary>
/// Options for CUDA graph capture
/// </summary>
public class CUDAGraphCaptureOptions
{
    /// <summary>
    /// Gets or sets the number of warm-up iterations
    /// </summary>
    public int WarmupIterations { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to validate the graph after capture
    /// </summary>
    public bool ValidateOnCapture { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable weight updates
    /// </summary>
    public bool EnableWeightUpdates { get; set; } = false;

    /// <summary>
    /// Gets or sets the memory pool to use
    /// </summary>
    public CUDAGraphMemoryPool? MemoryPool { get; set; } = null;

    /// <summary>
    /// Gets or sets the capture mode
    /// </summary>
    public CudaCaptureMode CaptureMode { get; set; } = CudaCaptureMode.CaptureModeThreadLocal;

    /// <summary>
    /// Gets default capture options
    /// </summary>
    public static CUDAGraphCaptureOptions Default => new CUDAGraphCaptureOptions();

    /// <summary>
    /// Sets the number of warm-up iterations
    /// </summary>
    /// <param name="iterations">Number of warm-up iterations</param>
    /// <returns>This options instance for method chaining</returns>
    public CUDAGraphCaptureOptions WithWarmup(int iterations)
    {
        if (iterations < 0)
            throw new ArgumentOutOfRangeException(nameof(iterations), "Warmup iterations must be non-negative");

        WarmupIterations = iterations;
        return this;
    }

    /// <summary>
    /// Sets whether to validate the graph after capture
    /// </summary>
    /// <param name="validate">Whether to validate</param>
    /// <returns>This options instance for method chaining</returns>
    public CUDAGraphCaptureOptions WithValidation(bool validate = true)
    {
        ValidateOnCapture = validate;
        return this;
    }

    /// <summary>
    /// Sets whether to enable weight updates
    /// </summary>
    /// <param name="enable">Whether to enable weight updates</param>
    /// <returns>This options instance for method chaining</returns>
    public CUDAGraphCaptureOptions WithWeightUpdates(bool enable = true)
    {
        EnableWeightUpdates = enable;
        return this;
    }

    /// <summary>
    /// Sets the memory pool to use
    /// </summary>
    /// <param name="pool">Memory pool</param>
    /// <returns>This options instance for method chaining</returns>
    public CUDAGraphCaptureOptions WithMemoryPool(CUDAGraphMemoryPool pool)
    {
        MemoryPool = pool ?? throw new ArgumentNullException(nameof(pool));
        return this;
    }

    /// <summary>
    /// Sets the capture mode
    /// </summary>
    /// <param name="mode">Capture mode</param>
    /// <returns>This options instance for method chaining</returns>
    public CUDAGraphCaptureOptions WithCaptureMode(CudaCaptureMode mode)
    {
        CaptureMode = mode;
        return this;
    }
}
