using RitterFramework.Core.Tensor;

namespace MLFramework.HAL;

/// <summary>
/// Represents a compute backend for a specific device type
/// </summary>
public interface IBackend
{
    /// <summary>
    /// Backend name (e.g., "MKL", "CUDA", "rocBLAS")
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Device type this backend supports
    /// </summary>
    DeviceType Type { get; }

    /// <summary>
    /// Check if this backend supports a specific operation
    /// </summary>
    bool SupportsOperation(Operation operation);

    /// <summary>
    /// Execute an operation on input tensors
    /// </summary>
    /// <param name="operation">Operation to execute</param>
    /// <param name="inputs">Input tensors</param>
    /// <returns>Result tensor</returns>
    Tensor ExecuteOperation(Operation operation, Tensor[] inputs);

    /// <summary>
    /// Initialize the backend (load libraries, etc.)
    /// </summary>
    void Initialize();

    /// <summary>
    /// Check if the backend is available (hardware present, libraries loaded)
    /// </summary>
    bool IsAvailable { get; }
}
