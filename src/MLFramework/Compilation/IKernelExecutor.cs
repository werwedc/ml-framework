using MLFramework.Fusion.Dynamic;
using Tensor = RitterFramework.Core.Tensor.Tensor;

namespace MLFramework.Compilation;

/// <summary>
/// Interface for executing compiled kernels
/// </summary>
public interface IKernelExecutor
{
    /// <summary>
    /// Executes a compiled kernel with the given inputs and outputs
    /// </summary>
    /// <param name="kernel">The compiled kernel to execute</param>
    /// <param name="inputs">Input tensors</param>
    /// <param name="outputs">Output tensors (pre-allocated)</param>
    void Execute(CompiledKernel kernel, List<Tensor> inputs, List<Tensor> outputs);
}
