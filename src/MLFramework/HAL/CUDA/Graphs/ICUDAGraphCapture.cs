namespace MLFramework.HAL.CUDA;

/// <summary>
/// Interface for graph capture operations
/// </summary>
public interface ICUDAGraphCapture : IDisposable
{
    /// <summary>
    /// Gets whether capture is currently active
    /// </summary>
    bool IsCapturing { get; }

    /// <summary>
    /// Begins capturing kernel launches
    /// </summary>
    /// <param name="stream">CUDA stream to capture from</param>
    void BeginCapture(CudaStream stream);

    /// <summary>
    /// Ends capture and returns the captured graph
    /// </summary>
    /// <returns>The captured CUDA graph</returns>
    ICUDAGraph EndCapture();
}
