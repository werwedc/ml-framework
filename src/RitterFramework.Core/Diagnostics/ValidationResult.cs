namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Represents the result of a shape validation operation.
/// </summary>
public class ValidationResult
{
    /// <summary>
    /// Indicates whether the validation passed.
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// List of error messages if validation failed.
    /// </summary>
    public List<string> Errors { get; set; }

    /// <summary>
    /// List of warning messages.
    /// </summary>
    public List<string> Warnings { get; set; }

    /// <summary>
    /// Creates a successful validation result.
    /// </summary>
    public static ValidationResult Success()
    {
        return new ValidationResult { IsValid = true };
    }

    /// <summary>
    /// Creates a failed validation result with the specified errors.
    /// </summary>
    public static ValidationResult Failure(params string[] errors)
    {
        return new ValidationResult
        {
            IsValid = false,
            Errors = new List<string>(errors)
        };
    }
}
