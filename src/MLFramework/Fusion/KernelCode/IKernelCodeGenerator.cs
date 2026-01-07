using MLFramework.Core;

namespace MLFramework.Fusion;

/// <summary>
/// Base interface for generating fused kernel code
/// </summary>
public interface IKernelCodeGenerator
{
    /// <summary>
    /// Generates kernel code for a fused operation
    /// </summary>
    KernelCodeResult GenerateKernel(FusedOperation fusedOp, GenerationOptions options);

    /// <summary>
    /// Gets supported backend type
    /// </summary>
    KernelBackendType BackendType { get; }

    /// <summary>
    /// Validates that a fused operation can be compiled by this backend
    /// </summary>
    bool CanCompile(FusedOperation fusedOp);
}

/// <summary>
/// Backend type for kernel generation
/// </summary>
public enum KernelBackendType
{
    CUDA,
    HIP,
    Triton,
    CUDAPlus,
    Metal,
    OpenCL
}

/// <summary>
/// Result of kernel code generation
/// </summary>
public record KernelCodeResult
{
    public required string KernelSourceCode { get; init; }
    public required string KernelName { get; init; }
    public required IReadOnlyList<KernelParameter> Parameters { get; init; }
    public required CompilationMetadata Metadata { get; init; }
    public required IReadOnlyList<string> Includes { get; init; }
}

/// <summary>
/// Parameter for kernel generation
/// </summary>
public record KernelParameter
{
    public required string Name { get; init; }
    public required ParameterDirection Direction { get; init; }
    public required DataType DataType { get; init; }
}

/// <summary>
/// Direction of parameter
/// </summary>
public enum ParameterDirection
{
    Input,
    Output,
    InputOutput
}

/// <summary>
/// Metadata for kernel compilation
/// </summary>
public record CompilationMetadata
{
    public required int SharedMemoryBytes { get; init; }
    public required int RegisterCount { get; init; }
    public required int ThreadBlockSize { get; init; }
    public required int GridSize { get; init; }
    public required IReadOnlySet<string> RequiredCapabilities { get; init; }
}

/// <summary>
/// Options for kernel code generation
/// </summary>
public record GenerationOptions
{
    public KernelBackendType TargetBackend { get; init; } = KernelBackendType.CUDA;
    public OptimizationLevel OptimizationLevel { get; init; } = OptimizationLevel.O3;
    public bool EnableVectorization { get; init; } = true;
    public bool EnableSharedMemory { get; init; } = true;
    public bool EnableUnrolling { get; init; } = true;
    public int ComputeCapability { get; init; } = 75; // SM_7.5
}

/// <summary>
/// Optimization level for code generation
/// </summary>
public enum OptimizationLevel
{
    None,
    O0,
    O1,
    O2,
    O3,
    Fast
}
