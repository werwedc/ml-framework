using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd;

/// <summary>
/// Represents the result of a gradient check operation.
/// Contains detailed information about the comparison between numerical and analytical gradients.
/// </summary>
public class GradientCheckResult
{
    /// <summary>
    /// Gets or sets whether the gradient check passed.
    /// A check passes when all gradient differences are within the specified tolerance.
    /// </summary>
    public bool Passed { get; set; }

    /// <summary>
    /// Gets or sets the maximum absolute difference found between numerical and analytical gradients.
    /// </summary>
    public double MaxAbsoluteDifference { get; set; }

    /// <summary>
    /// Gets or sets the maximum relative error found between numerical and analytical gradients.
    /// </summary>
    public double MaxRelativeError { get; set; }

    /// <summary>
    /// Gets or sets a list of tensor differences that exceeded the tolerance threshold.
    /// Empty list if all differences are within tolerance.
    /// </summary>
    public List<TensorDifference> Differences { get; set; } = new List<TensorDifference>();

    /// <summary>
    /// Gets or sets a description of why the gradient check failed.
    /// Null if the check passed.
    /// </summary>
    public string? FailureReason { get; set; }

    /// <summary>
    /// Returns a summary string of the gradient check result.
    /// </summary>
    /// <returns>A human-readable summary of the result.</returns>
    public string GetSummary()
    {
        var status = Passed ? "PASSED" : "FAILED";
        var summary = $"Gradient Check: {status}\n";
        summary += $"Max Absolute Difference: {MaxAbsoluteDifference:E6}\n";
        summary += $"Max Relative Error: {MaxRelativeError:E6}\n";

        if (Differences.Count > 0)
        {
            summary += $"\n{Differences.Count} difference(s) found (showing first 10):\n";
            var displayCount = Math.Min(10, Differences.Count);
            for (int i = 0; i < displayCount; i++)
            {
                var diff = Differences[i];
                summary += $"  Input[{diff.InputIndex}][{string.Join(",", diff.ElementIndex)}]: " +
                          $"num={diff.NumericalValue:E6}, ana={diff.AnalyticalValue:E6}, " +
                          $"abs={diff.AbsoluteDifference:E6}, rel={diff.RelativeError:E6}\n";
            }
            if (Differences.Count > 10)
            {
                summary += $"  ... and {Differences.Count - 10} more differences\n";
            }
        }

        if (!Passed && FailureReason != null)
        {
            summary += $"\nReason: {FailureReason}";
        }

        return summary;
    }
}

/// <summary>
/// Represents a difference between numerical and analytical gradient values at a specific tensor element.
/// </summary>
public class TensorDifference
{
    /// <summary>
    /// Gets or sets the index of the input tensor that contains this difference.
    /// </summary>
    public int InputIndex { get; set; }

    /// <summary>
    /// Gets or sets the multi-dimensional index of the element within the tensor.
    /// </summary>
    public int[] ElementIndex { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets the numerically computed gradient value.
    /// </summary>
    public double NumericalValue { get; set; }

    /// <summary>
    /// Gets or sets the analytically computed gradient value.
    /// </summary>
    public double AnalyticalValue { get; set; }

    /// <summary>
    /// Gets or sets the absolute difference between numerical and analytical values.
    /// </summary>
    public double AbsoluteDifference { get; set; }

    /// <summary>
    /// Gets or sets the relative error between numerical and analytical values.
    /// </summary>
    public double RelativeError { get; set; }
}
