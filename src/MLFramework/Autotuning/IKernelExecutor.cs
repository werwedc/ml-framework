using MLFramework.Fusion;
using Backends = MLFramework.Fusion.Backends;
using RitterFramework.Core;
using Tensor = RitterFramework.Core.Tensor.Tensor;

namespace MLFramework.Autotuning;

/// <summary>
/// Interface for executing fused kernels
/// </summary>
public interface IKernelExecutor
{
    /// <summary>
    /// Executes a fused kernel operation
    /// </summary>
    void ExecuteFusedKernel(MLFramework.Fusion.FusedOperation fusedOp, Backends.KernelLaunchConfiguration config, Tensor input);

    /// <summary>
    /// Executes a single kernel operation
    /// </summary>
    Tensor ExecuteKernel(Operation op, Tensor input);

    /// <summary>
    /// Synchronizes device execution
    /// </summary>
    void Synchronize();
}
