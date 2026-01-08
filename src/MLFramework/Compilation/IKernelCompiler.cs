using MLFramework.Fusion;
using MLFramework.Fusion.Dynamic;

namespace MLFramework.Compilation;

/// <summary>
/// Interface for compiling kernels
/// </summary>
public interface IKernelCompiler
{
    /// <summary>
    /// Compiles a kernel for the given operation and shapes
    /// </summary>
    /// <param name="op">The operation to compile</param>
    /// <param name="inputShapes">Input tensor shapes</param>
    /// <param name="outputShapes">Output tensor shapes</param>
    /// <returns>A compiled kernel</returns>
    CompiledKernel Compile(Operation op, List<int[]> inputShapes, List<int[]> outputShapes);

    /// <summary>
    /// Checks if this compiler can compile the given operation
    /// </summary>
    /// <param name="op">The operation to check</param>
    /// <returns>True if the operation can be compiled</returns>
    bool CanCompile(Operation op);
}
