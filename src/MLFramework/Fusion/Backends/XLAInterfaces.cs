namespace MLFramework.Fusion.Backends;

/// <summary>
/// Interface for XLA compiler
/// </summary>
public interface IXLACompiler
{
    /// <summary>
    /// Compiles an XLA-fused operation
    /// </summary>
    /// <param name="fusedOp">Fused operation to compile</param>
    /// <param name="options">Compilation options</param>
    /// <returns>Compiled kernel binary</returns>
    KernelBinary CompileXLAFused(FusedOperation fusedOp, CompilationOptions options);
}

/// <summary>
/// Mock XLA compiler for testing
/// </summary>
public class MockXLACompiler : IXLACompiler
{
    public KernelBinary CompileXLAFused(FusedOperation fusedOp, CompilationOptions options)
    {
        // Mock implementation - returns a dummy binary
        return new KernelBinary
        {
            Data = Array.Empty<byte>(),
            Format = "hlo",
            Metadata = new Dictionary<string, object>
            {
                { "kernelName", fusedOp.KernelSpec.KernelName },
                { "optimizationLevel", options.OptimizationLevel }
            }
        };
    }
}
