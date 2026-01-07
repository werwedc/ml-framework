using RitterFramework.Core;
using Tensor = RitterFramework.Core.Tensor.Tensor;
using MLFramework.Fusion.Backends;

namespace MLFramework.Fusion.Validation;

/// <summary>
/// Interface for verifying fused operations produce correct results
/// </summary>
public interface IFusionVerifier
{
    /// <summary>
    /// Verifies that fused operation produces same result as sequential ops
    /// </summary>
    VerificationResult Verify(
        FusedOperation fusedOp,
        IReadOnlyList<Operation> originalOps,
        Tensor testInput);

    /// <summary>
    /// Runs verification on random test inputs
    /// </summary>
    VerificationResult VerifyWithRandomInputs(
        FusedOperation fusedOp,
        IReadOnlyList<Operation> originalOps,
        int testCases = 10);
}
