namespace MLFramework.Utilities;

using System.Collections.Generic;

/// <summary>
/// Validation result for deterministic behavior
/// </summary>
public class ValidationResult
{
    /// <summary>
    /// Whether validation passed
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// List of validation messages
    /// </summary>
    public List<string> Messages { get; set; }

    /// <summary>
    /// List of warnings
    /// </summary>
    public List<string> Warnings { get; set; }

    /// <summary>
    /// List of errors
    /// </summary>
    public List<string> Errors { get; set; }

    public ValidationResult()
    {
        Messages = new List<string>();
        Warnings = new List<string>();
        Errors = new List<string>();
    }

    public bool HasWarnings => Warnings.Count > 0;
    public bool HasErrors => Errors.Count > 0;
}
