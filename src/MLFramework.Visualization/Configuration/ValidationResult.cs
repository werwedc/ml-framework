namespace MLFramework.Visualization.Configuration;

/// <summary>
/// Result of configuration validation
/// </summary>
public class ValidationResult
{
    /// <summary>
    /// Whether the configuration is valid
    /// </summary>
    public bool IsValid { get; private set; }

    /// <summary>
    /// List of validation errors
    /// </summary>
    public List<string> Errors { get; private set; }

    /// <summary>
    /// List of validation warnings
    /// </summary>
    public List<string> Warnings { get; private set; }

    /// <summary>
    /// Creates a successful validation result
    /// </summary>
    public ValidationResult()
    {
        IsValid = true;
        Errors = new List<string>();
        Warnings = new List<string>();
    }

    /// <summary>
    /// Creates a validation result with errors
    /// </summary>
    public ValidationResult(List<string> errors, List<string> warnings = null)
    {
        IsValid = errors == null || errors.Count == 0;
        Errors = errors ?? new List<string>();
        Warnings = warnings ?? new List<string>();
    }

    /// <summary>
    /// Add an error to the validation result
    /// </summary>
    public void AddError(string error)
    {
        Errors.Add(error);
        IsValid = false;
    }

    /// <summary>
    /// Add a warning to the validation result
    /// </summary>
    public void AddWarning(string warning)
    {
        Warnings.Add(warning);
    }

    /// <summary>
    /// Get a formatted string of all errors and warnings
    /// </summary>
    public string GetSummary()
    {
        var summary = new List<string>();

        if (Errors.Count > 0)
        {
            summary.Add("Errors:");
            summary.AddRange(Errors.Select(e => $"  - {e}"));
        }

        if (Warnings.Count > 0)
        {
            summary.Add("Warnings:");
            summary.AddRange(Warnings.Select(w => $"  - {w}"));
        }

        if (summary.Count == 0)
        {
            return "Configuration is valid.";
        }

        return string.Join("\n", summary);
    }
}
