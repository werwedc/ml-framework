namespace MLFramework.Fusion.Dynamic;

/// <summary>
/// Represents a compiled kernel that can be executed
/// </summary>
public class CompiledKernel
{
    /// <summary>
    /// Gets the unique identifier for this compiled kernel
    /// </summary>
    public required string KernelId { get; init; }

    /// <summary>
    /// Gets the source code for this kernel
    /// </summary>
    public required string SourceCode { get; init; }

    /// <summary>
    /// Gets the compiled binary (bytecode)
    /// </summary>
    public required byte[] Binary { get; init; }

    /// <summary>
    /// Gets the shapes this kernel is specialized for
    /// </summary>
    public required IReadOnlyList<int[]> SpecializedShapes { get; init; }

    /// <summary>
    /// Gets whether this is a generic kernel
    /// </summary>
    public required bool IsGeneric { get; init; }

    /// <summary>
    /// Gets the signature of this kernel
    /// </summary>
    public required string Signature { get; init; }

    /// <summary>
    /// Gets the estimated execution time for this kernel
    /// </summary>
    public required long EstimatedExecutionTimeNs { get; init; }
}
