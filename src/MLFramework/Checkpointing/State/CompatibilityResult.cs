namespace MachineLearning.Checkpointing;

/// <summary>
/// Result of a compatibility check between state dictionaries
/// </summary>
public class CompatibilityResult
{
    /// <summary>
    /// List of errors found during compatibility check
    /// </summary>
    public List<string> Errors { get; } = new();

    /// <summary>
    /// List of warnings found during compatibility check
    /// </summary>
    public List<string> Warnings { get; } = new();

    /// <summary>
    /// Whether the states are compatible (no errors)
    /// </summary>
    public bool IsCompatible => Errors.Count == 0;

    /// <summary>
    /// Whether there were any warnings
    /// </summary>
    public bool HasWarnings => Warnings.Count > 0;

    /// <summary>
    /// Total number of issues (errors + warnings)
    /// </summary>
    public int TotalIssues => Errors.Count + Warnings.Count;

    /// <summary>
    /// Adds an error to the result
    /// </summary>
    /// <param name="error">Error message</param>
    public void AddError(string error)
    {
        if (!string.IsNullOrWhiteSpace(error))
        {
            Errors.Add(error);
        }
    }

    /// <summary>
    /// Adds a warning to the result
    /// </summary>
    /// <param name="warning">Warning message</param>
    public void AddWarning(string warning)
    {
        if (!string.IsNullOrWhiteSpace(warning))
        {
            Warnings.Add(warning);
        }
    }

    /// <summary>
    /// Adds multiple errors to the result
    /// </summary>
    /// <param name="errors">Collection of error messages</param>
    public void AddErrors(IEnumerable<string> errors)
    {
        if (errors != null)
        {
            foreach (var error in errors.Where(e => !string.IsNullOrWhiteSpace(e)))
            {
                Errors.Add(error);
            }
        }
    }

    /// <summary>
    /// Adds multiple warnings to the result
    /// </summary>
    /// <param name="warnings">Collection of warning messages</param>
    public void AddWarnings(IEnumerable<string> warnings)
    {
        if (warnings != null)
        {
            foreach (var warning in warnings.Where(w => !string.IsNullOrWhiteSpace(w)))
            {
                Warnings.Add(warning);
            }
        }
    }

    /// <summary>
    /// Merges another compatibility result into this one
    /// </summary>
    /// <param name="other">Other compatibility result to merge</param>
    public void Merge(CompatibilityResult other)
    {
        if (other != null)
        {
            AddErrors(other.Errors);
            AddWarnings(other.Warnings);
        }
    }

    /// <summary>
    /// Creates a successful compatibility result
    /// </summary>
    public static CompatibilityResult Success()
    {
        return new CompatibilityResult();
    }

    /// <summary>
    /// Creates a failed compatibility result with a single error
    /// </summary>
    /// <param name="error">Error message</param>
    /// <returns>Compatibility result with the error</returns>
    public static CompatibilityResult Failure(string error)
    {
        var result = new CompatibilityResult();
        result.AddError(error);
        return result;
    }

    /// <summary>
    /// Creates a compatibility result with multiple errors
    /// </summary>
    /// <param name="errors">Collection of error messages</param>
    /// <returns>Compatibility result with the errors</returns>
    public static CompatibilityResult Failures(IEnumerable<string> errors)
    {
        var result = new CompatibilityResult();
        result.AddErrors(errors);
        return result;
    }

    /// <summary>
    /// Gets a formatted summary of the compatibility result
    /// </summary>
    /// <returns>String representation of the result</returns>
    public override string ToString()
    {
        var summary = new System.Text.StringBuilder();
        summary.AppendLine($"Compatibility Result: {(IsCompatible ? "COMPATIBLE" : "INCOMPATIBLE")}");
        summary.AppendLine($"Errors: {Errors.Count}, Warnings: {Warnings.Count}");

        if (Errors.Count > 0)
        {
            summary.AppendLine();
            summary.AppendLine("Errors:");
            foreach (var error in Errors)
            {
                summary.AppendLine($"  - {error}");
            }
        }

        if (Warnings.Count > 0)
        {
            summary.AppendLine();
            summary.AppendLine("Warnings:");
            foreach (var warning in Warnings)
            {
                summary.AppendLine($"  - {warning}");
            }
        }

        return summary.ToString();
    }
}
