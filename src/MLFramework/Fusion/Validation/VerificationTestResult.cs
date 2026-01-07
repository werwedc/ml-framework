using RitterFramework.Core;
using Tensor = RitterFramework.Core.Tensor.Tensor;

namespace MLFramework.Fusion.Validation;

/// <summary>
/// Result of a single verification test case
/// </summary>
public record VerificationTestResult
{
    /// <summary>
    /// Test case number
    /// </summary>
    public required int TestCaseNumber { get; init; }

    /// <summary>
    /// Output from fused operation
    /// </summary>
    public required Tensor FusedOutput { get; init; }

    /// <summary>
    /// Output from sequential operations
    /// </summary>
    public required Tensor SequentialOutput { get; init; }

    /// <summary>
    /// Error between outputs
    /// </summary>
    public required double Error { get; init; }

    /// <summary>
    /// Whether the error is within acceptable tolerance
    /// </summary>
    public required bool TolerancePassed { get; init; }
}
