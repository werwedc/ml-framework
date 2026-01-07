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
    void ExecuteFusedKernel(MLFramework.Fusion.FusedOperation fusedOp, Backends.KernelLaunchConfiguration config, Tensor input);
    void Synchronize();
}
