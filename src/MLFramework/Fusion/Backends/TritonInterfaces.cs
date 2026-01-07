namespace MLFramework.Fusion.Backends;

/// <summary>
/// Interface for Triton compiler
/// </summary>
public interface ITritonCompiler
{
    /// <summary>
    /// Compiles a fused operation into a Triton kernel binary
    /// </summary>
    /// <param name="fusedOp">Fused operation to compile</param>
    /// <param name="options">Compilation options</param>
    /// <returns>Compiled kernel binary</returns>
    KernelBinary Compile(FusedOperation fusedOp, CompilationOptions options);
}

/// <summary>
/// Interface for Triton autotuner
/// </summary>
public interface ITritonAutotuner
{
    /// <summary>
    /// Tunes the fused operation for optimal performance
    /// </summary>
    /// <param name="fusedOp">Fused operation to tune</param>
    /// <returns>Tuned fused operation</returns>
    FusedOperation Tune(FusedOperation fusedOp);

    /// <summary>
    /// Gets the best launch configuration for a fused operation
    /// </summary>
    /// <param name="fusedOp">Fused operation</param>
    /// <returns>Best launch configuration</returns>
    KernelLaunchConfiguration GetBestLaunchConfig(FusedOperation fusedOp);
}

/// <summary>
/// Mock Triton compiler for testing
/// </summary>
public class MockTritonCompiler : ITritonCompiler
{
    public KernelBinary Compile(FusedOperation fusedOp, CompilationOptions options)
    {
        // Mock implementation - returns a dummy binary
        return new KernelBinary
        {
            Data = Array.Empty<byte>(),
            Format = "ptx",
            Metadata = new Dictionary<string, object>
            {
                { "kernelName", fusedOp.KernelSpec.KernelName },
                { "optimizationLevel", options.OptimizationLevel }
            }
        };
    }
}

/// <summary>
/// Mock Triton autotuner for testing
/// </summary>
public class MockTritonAutotuner : ITritonAutotuner
{
    public FusedOperation Tune(FusedOperation fusedOp)
    {
        // Mock implementation - returns the same operation
        return fusedOp;
    }

    public KernelLaunchConfiguration GetBestLaunchConfig(FusedOperation fusedOp)
    {
        // Mock implementation - returns a default config
        return new KernelLaunchConfiguration
        {
            BlockDim = new ThreadBlockConfiguration { X = 128, Y = 1, Z = 1 },
            GridDim = new ThreadBlockConfiguration { X = 1, Y = 1, Z = 1 },
            SharedMemoryBytes = 0,
            Parameters = fusedOp.KernelSpec.Parameters.ToList()
        };
    }
}
