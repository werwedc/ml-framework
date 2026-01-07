using RitterFramework.Core;
using Tensor = RitterFramework.Core.Tensor.Tensor;

namespace MLFramework.Fusion.Validation;

/// <summary>
/// Interface for handling fusion fallback when operations cannot be fused
/// </summary>
public interface IFusionFallback
{
    /// <summary>
    /// Attempts to execute operations as separate kernels
    /// </summary>
    Tensor ExecuteSeparate(IReadOnlyList<Operation> operations, Tensor input);

    /// <summary>
    /// Logs the reason for falling back
    /// </summary>
    void LogFallbackReason(string reason, IReadOnlyList<Operation> operations);
}
