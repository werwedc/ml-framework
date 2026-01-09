namespace MLFramework.Utilities;

/// <summary>
/// Flags for controlling deterministic mode settings in the ML framework.
/// These flags enable deterministic algorithms in various backends (cuDNN, cuBLAS, CUDA) to ensure reproducible computations.
/// </summary>
[Flags]
public enum DeterministicModeFlags
{
    /// <summary>
    /// No deterministic mode enabled
    /// </summary>
    None = 0,

    /// <summary>
    /// Enable deterministic algorithms in cuDNN.
    /// This forces cuDNN to use deterministic convolution algorithms, which may impact performance by 10-30%.
    /// </summary>
    CudnnDeterministic = 1 << 0,

    /// <summary>
    /// Enable deterministic algorithms in cuBLAS.
    /// Configures cuBLAS to use PEDANTIC mode, which may impact GEMM operations by 15-25%.
    /// </summary>
    CublasDeterministic = 1 << 1,

    /// <summary>
    /// Enable deterministic memory allocation in CUDA.
    /// Sets memory pool size limit for deterministic allocation using cudaDeviceSetLimit.
    /// </summary>
    CudaMemoryDeterministic = 1 << 2,

    /// <summary>
    /// Enable CUDA Graphs for deterministic kernel launch ordering.
    /// Ensures deterministic kernel launch ordering, may improve performance for repeated operations.
    /// </summary>
    CudaGraphs = 1 << 3,

    /// <summary>
    /// Enable all deterministic modes.
    /// Combines all deterministic flags for maximum reproducibility.
    /// </summary>
    All = CudnnDeterministic | CublasDeterministic | CudaMemoryDeterministic | CudaGraphs
}
