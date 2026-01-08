namespace MLFramework.HAL.CUDA;

/// <summary>
/// Interface for CUDA graph operations
/// </summary>
public interface ICUDAGraph : IDisposable
{
    /// <summary>
    /// Gets the unique identifier for this graph
    /// </summary>
    string GraphId { get; }

    /// <summary>
    /// Gets the current state of the graph
    /// </summary>
    CUDAGraphState State { get; }

    /// <summary>
    /// Executes the captured graph
    /// </summary>
    /// <param name="stream">CUDA stream for execution</param>
    void Execute(CudaStream stream);

    /// <summary>
    /// Validates that all captured operations are graph-compatible
    /// </summary>
    /// <returns>Validation result with any errors</returns>
    CUDAGraphValidationResult Validate();
}
