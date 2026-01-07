using RitterFramework.Core;
using Tensor = RitterFramework.Core.Tensor.Tensor;

namespace MLFramework.Fusion.Validation;

/// <summary>
/// Result of a fusion verification test
/// </summary>
public record VerificationResult
{
    /// <summary>
    /// Whether the verification passed (all test cases within tolerance)
    /// </summary>
    public required bool Passed { get; init; }

    /// <summary>
    /// Maximum error across all test cases
    /// </summary>
    public required double MaxError { get; init; }

    /// <summary>
    /// Mean error across all test cases
    /// </summary>
    public required double MeanError { get; init; }

    /// <summary>
    /// Individual test case results
    /// </summary>
    public required IReadOnlyList<VerificationTestResult> TestCases { get; init; }

    /// <summary>
    /// Number of test cases that failed
    /// </summary>
    public int FailedTestCases => TestCases.Count(tc => !tc.TolerancePassed);

    /// <summary>
    /// Number of test cases that passed
    /// </summary>
    public int PassedTestCases => TestCases.Count(tc => tc.TolerancePassed);
}
