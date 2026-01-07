using MLFramework.Core;

namespace MLFramework.Fusion.Backends;

/// <summary>
/// Type of fusion backend
/// </summary>
public enum FusionBackendType
{
    /// <summary>Triton GPU kernel compiler</summary>
    Triton,
    /// <summary>XLA (Accelerated Linear Algebra) compiler</summary>
    XLA,
    /// <summary>Custom fusion backend</summary>
    Custom,
    /// <summary>No backend</summary>
    None
}

/// <summary>
/// Types of fusion patterns
/// </summary>
public enum FusionPatternType
{
    /// <summary>Element-wise operations</summary>
    ElementWise,
    /// <summary>Convolution followed by activation</summary>
    ConvActivation,
    /// <summary>Convolution followed by batch normalization</summary>
    ConvBatchNorm,
    /// <summary>Linear followed by activation</summary>
    LinearActivation,
    /// <summary>Reduction followed by element-wise operation</summary>
    ReductionThenElementWise,
    /// <summary>Mixed pattern type</summary>
    Mixed
}

/// <summary>
/// Memory location for fusion variables
/// </summary>
public enum MemoryLocation
{
    /// <summary>Input tensor</summary>
    Input,
    /// <summary>Output tensor</summary>
    Output,
    /// <summary>Temporary/intermediate tensor</summary>
    Temporary
}

/// <summary>
/// Compiled kernel with binary and metadata
/// </summary>
public record CompiledKernel
{
    /// <summary>
    /// Unique identifier for the compiled kernel
    /// </summary>
    public required string KernelId { get; init; }

    /// <summary>
    /// The fused operation this kernel represents
    /// </summary>
    public required FusedOperation Operation { get; init; }

    /// <summary>
    /// The compiled binary code
    /// </summary>
    public required KernelBinary Binary { get; init; }

    /// <summary>
    /// Launch configuration for the kernel
    /// </summary>
    public required KernelLaunchConfiguration LaunchConfig { get; init; }

    /// <summary>
    /// Compilation metrics
    /// </summary>
    public required CompilationMetrics Metrics { get; init; }
}

/// <summary>
/// Binary representation of a compiled kernel
/// </summary>
public record KernelBinary
{
    /// <summary>
    /// Raw binary data
    /// </summary>
    public required byte[] Data { get; init; }

    /// <summary>
    /// Binary format (e.g., "ptx", "cubin", "hlo")
    /// </summary>
    public required string Format { get; init; }

    /// <summary>
    /// Metadata about the binary
    /// </summary>
    public required IReadOnlyDictionary<string, object> Metadata { get; init; }
}

/// <summary>
/// Metrics from kernel compilation
/// </summary>
public record CompilationMetrics
{
    /// <summary>
    /// Time taken to compile in milliseconds
    /// </summary>
    public required double CompilationTimeMs { get; init; }

    /// <summary>
    /// Size of the compiled binary in bytes
    /// </summary>
    public required long BinarySizeBytes { get; init; }

    /// <summary>
    /// Optimization level used during compilation
    /// </summary>
    public required int OptimizationLevel { get; init; }
}

/// <summary>
/// Configuration for backend initialization
/// </summary>
public record BackendConfiguration
{
    /// <summary>
    /// Device identifier
    /// </summary>
    public required string DeviceId { get; init; }

    /// <summary>
    /// Backend-specific options
    /// </summary>
    public required Dictionary<string, object> Options { get; init; } = new();
}

/// <summary>
/// Capabilities of a fusion backend
/// </summary>
public record BackendCapabilities
{
    /// <summary>
    /// Supported fusion patterns
    /// </summary>
    public required IReadOnlySet<FusionPatternType> SupportedPatterns { get; init; }

    /// <summary>
    /// Supported tensor data types
    /// </summary>
    public required IReadOnlySet<DataType> SupportedDataTypes { get; init; }

    /// <summary>
    /// Whether backend supports autotuning
    /// </summary>
    public required bool SupportsAutotuning { get; init; }

    /// <summary>
    /// Whether backend supports profiling
    /// </summary>
    public required bool SupportsProfiling { get; init; }

    /// <summary>
    /// Whether backend supports JIT compilation
    /// </summary>
    public required bool SupportsJITCompilation { get; init; }

    /// <summary>
    /// Whether backend supports binary caching
    /// </summary>
    public required bool SupportsBinaryCache { get; init; }
}

/// <summary>
/// Options for fusion operation
/// </summary>
public record FusionOptions
{
    /// <summary>
    /// Maximum number of operations to fuse
    /// </summary>
    public int MaxOpsToFuse { get; init; } = 10;

    /// <summary>
    /// Whether to validate fused operations
    /// </summary>
    public bool ValidateFusions { get; init; } = true;

    /// <summary>
    /// Additional fusion options
    /// </summary>
    public Dictionary<string, object> AdditionalOptions { get; init; } = new();
}

/// <summary>
/// Options for kernel compilation
/// </summary>
public record CompilationOptions
{
    /// <summary>
    /// Optimization level (0-3)
    /// </summary>
    public int OptimizationLevel { get; init; } = 2;

    /// <summary>
    /// Whether to use debug mode
    /// </summary>
    public bool DebugMode { get; init; } = false;

    /// <summary>
    /// Additional compilation flags
    /// </summary>
    public IReadOnlyList<string> CompilationFlags { get; init; } = Array.Empty<string>();
}

/// <summary>
/// Result of a fusion operation
/// </summary>
public record FusionResult
{
    /// <summary>
    /// Computational graph with fused operations
    /// </summary>
    public required ComputationalGraph FusedGraph { get; init; }

    /// <summary>
    /// List of fused operations
    /// </summary>
    public required IReadOnlyList<FusedOperation> FusedOperations { get; init; }

    /// <summary>
    /// Number of original operations
    /// </summary>
    public required int OriginalOpCount { get; init; }

    /// <summary>
    /// Number of fused operations
    /// </summary>
    public required int FusedOpCount { get; init; }

    /// <summary>
    /// List of rejected fusion attempts
    /// </summary>
    public required IReadOnlyList<FusionRejected> RejectedFusions { get; init; }
}

/// <summary>
/// Information about a rejected fusion
/// </summary>
public record FusionRejected
{
    /// <summary>
    /// Operations that were attempted to be fused
    /// </summary>
    public required IReadOnlyList<Operation> Operations { get; init; }

    /// <summary>
    /// Reason for rejection
    /// </summary>
    public required string Reason { get; init; }
}

/// <summary>
/// Kernel launch configuration
/// </summary>
public record KernelLaunchConfiguration
{
    /// <summary>
    /// Block dimension (thread block size)
    /// </summary>
    public required ThreadBlockConfiguration BlockDim { get; init; }

    /// <summary>
    /// Grid dimension (number of blocks)
    /// </summary>
    public required ThreadBlockConfiguration GridDim { get; init; }

    /// <summary>
    /// Shared memory in bytes
    /// </summary>
    public required int SharedMemoryBytes { get; init; }

    /// <summary>
    /// Kernel launch parameters
    /// </summary>
    public required IReadOnlyList<KernelLaunchParameter> Parameters { get; init; }
}

/// <summary>
/// Thread block configuration
/// </summary>
public record ThreadBlockConfiguration
{
    /// <summary>
    /// X dimension
    /// </summary>
    public int X { get; init; } = 1;

    /// <summary>
    /// Y dimension
    /// </summary>
    public int Y { get; init; } = 1;

    /// <summary>
    /// Z dimension
    /// </summary>
    public int Z { get; init; } = 1;
}

/// <summary>
/// Kernel launch parameter
/// </summary>
public record KernelLaunchParameter
{
    /// <summary>
    /// Parameter name
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Parameter value (null for input/output tensors)
    /// </summary>
    public object? Value { get; init; }

    /// <summary>
    /// Parameter type
    /// </summary>
    public required Type Type { get; init; }
}

/// <summary>
/// Fused operation representation
/// </summary>
public record FusedOperation : Operation
{
    /// <summary>
    /// Constituent operations that were fused
    /// </summary>
    public required IReadOnlyList<Operation> ConstituentOperations { get; init; }

    /// <summary>
    /// Fusion pattern used
    /// </summary>
    public required FusionPatternDefinition Pattern { get; init; }

    /// <summary>
    /// Intermediate representation for the fused operation
    /// </summary>
    public required FusionIR IR { get; init; }

    /// <summary>
    /// Kernel specification for the fused operation
    /// </summary>
    public required KernelSpecification KernelSpec { get; init; }
}

/// <summary>
/// Intermediate representation for fused operations
/// </summary>
public record FusionIR
{
    /// <summary>
    /// Unique identifier for this IR
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Nodes in the fusion graph
    /// </summary>
    public required IReadOnlyList<FusionOpNode> Nodes { get; init; }

    /// <summary>
    /// Variables used in the fusion
    /// </summary>
    public required IReadOnlyList<FusionVariable> Variables { get; init; }

    /// <summary>
    /// Memory layout information
    /// </summary>
    public required MemoryLayout MemoryLayout { get; init; }

    /// <summary>
    /// Compute requirements
    /// </summary>
    public required ComputeRequirements ComputeRequirements { get; init; }
}

/// <summary>
/// Node in fusion IR graph
/// </summary>
public record FusionOpNode
{
    /// <summary>
    /// Node identifier
    /// </summary>
    public required string Id { get; init; }

    /// <summary>
    /// Original operation type
    /// </summary>
    public required string OriginalOpType { get; init; }

    /// <summary>
    /// Input variable names
    /// </summary>
    public required IReadOnlyList<string> InputVars { get; init; }

    /// <summary>
    /// Output variable name
    /// </summary>
    public required string OutputVar { get; init; }

    /// <summary>
    /// Node attributes
    /// </summary>
    public required IReadOnlyDictionary<string, object> Attributes { get; init; }
}

/// <summary>
/// Variable in fusion IR
/// </summary>
public record FusionVariable
{
    /// <summary>
    /// Variable name
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Tensor shape
    /// </summary>
    public required TensorShape Shape { get; init; }

    /// <summary>
    /// Data type
    /// </summary>
    public required DataType DataType { get; init; }

    /// <summary>
    /// Memory location
    /// </summary>
    public required MemoryLocation Location { get; init; }
}

/// <summary>
/// Memory layout information
/// </summary>
public record MemoryLayout
{
    /// <summary>
    /// Tensor layout
    /// </summary>
    public required TensorLayout TensorLayout { get; init; }

    /// <summary>
    /// Shared memory in bytes
    /// </summary>
    public required int SharedMemoryBytes { get; init; }

    /// <summary>
    /// Register usage in bytes
    /// </summary>
    public required int RegisterBytes { get; init; }
}

/// <summary>
/// Compute requirements
/// </summary>
public record ComputeRequirements
{
    /// <summary>
    /// Number of thread blocks
    /// </summary>
    public required int ThreadBlocks { get; init; }

    /// <summary>
    /// Threads per block
    /// </summary>
    public required int ThreadsPerBlock { get; init; }

    /// <summary>
    /// Whether shared memory is required
    /// </summary>
    public required bool RequiresSharedMemory { get; init; }

    /// <summary>
    /// Whether atomic operations are required
    /// </summary>
    public required bool RequiresAtomicOps { get; init; }
}

/// <summary>
/// Kernel specification
/// </summary>
public record KernelSpecification
{
    /// <summary>
    /// Kernel name
    /// </summary>
    public required string KernelName { get; init; }

    /// <summary>
    /// Fusion strategy
    /// </summary>
    public required FusionStrategy Strategy { get; init; }

    /// <summary>
    /// Input tensors
    /// </summary>
    public required IReadOnlyList<FusionVariable> InputTensors { get; init; }

    /// <summary>
    /// Output tensors
    /// </summary>
    public required IReadOnlyList<FusionVariable> OutputTensors { get; init; }

    /// <summary>
    /// Temporary memory required in bytes
    /// </summary>
    public required int TemporaryMemoryBytes { get; init; }

    /// <summary>
    /// Register usage in bytes
    /// </summary>
    public required int RegisterBytes { get; init; }

    /// <summary>
    /// Thread block configuration
    /// </summary>
    public required ThreadBlockConfiguration ThreadBlockConfig { get; init; }

    /// <summary>
    /// Compilation flags
    /// </summary>
    public required IReadOnlyList<string> CompilationFlags { get; init; }

    /// <summary>
    /// Kernel parameters
    /// </summary>
    public required IReadOnlyList<KernelLaunchParameter> Parameters { get; init; } = Array.Empty<KernelLaunchParameter>();
}

/// <summary>
/// Computational graph
/// </summary>
public record ComputationalGraph
{
    /// <summary>
    /// Operations in the graph
    /// </summary>
    public required IReadOnlyList<Operation> Operations { get; init; }

    /// <summary>
    /// Input tensors
    /// </summary>
    public required IReadOnlyList<string> Inputs { get; init; }

    /// <summary>
    /// Output tensors
    /// </summary>
    public required IReadOnlyList<string> Outputs { get; init; }
}
