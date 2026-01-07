namespace MLFramework.Fusion.Backends;

/// <summary>
/// Interface for fusion backends
/// </summary>
public interface IFusionBackend
{
    /// <summary>
    /// Gets the backend name
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the backend type
    /// </summary>
    FusionBackendType Type { get; }

    /// <summary>
    /// Checks if this backend can fuse the given operations
    /// </summary>
    /// <param name="operations">Operations to check</param>
    /// <returns>True if the backend can fuse the operations</returns>
    bool CanFuse(IReadOnlyList<Operation> operations);

    /// <summary>
    /// Fuses the given operations
    /// </summary>
    /// <param name="operations">Operations to fuse</param>
    /// <param name="options">Fusion options</param>
    /// <returns>Fusion result</returns>
    FusionResult Fuse(IReadOnlyList<Operation> operations, FusionOptions options);

    /// <summary>
    /// Compiles a fused operation into executable kernel
    /// </summary>
    /// <param name="fusedOp">Fused operation to compile</param>
    /// <param name="options">Compilation options</param>
    /// <returns>Compiled kernel</returns>
    CompiledKernel Compile(FusedOperation fusedOp, CompilationOptions options);

    /// <summary>
    /// Gets backend-specific capabilities
    /// </summary>
    /// <returns>Backend capabilities</returns>
    BackendCapabilities GetCapabilities();

    /// <summary>
    /// Initializes the backend
    /// </summary>
    /// <param name="config">Backend configuration</param>
    void Initialize(BackendConfiguration config);
}
