namespace RitterFramework.Core;

/// <summary>
/// Represents the result of a validation operation with optional diagnostic information.
/// </summary>
public struct ValidationResult
{
    /// <summary>
    /// Gets whether the validation passed.
    /// </summary>
    public bool IsValid { get; }

    /// <summary>
    /// Gets the error message if validation failed.
    /// </summary>
    public string ErrorMessage { get; }

    /// <summary>
    /// Gets a list of suggested fixes for the validation issue.
    /// </summary>
    public List<string> SuggestedFixes { get; }

    /// <summary>
    /// Creates a valid validation result.
    /// </summary>
    public static ValidationResult Valid()
    {
        return new ValidationResult(true, string.Empty, new List<string>());
    }

    /// <summary>
    /// Creates an invalid validation result with an error message.
    /// </summary>
    /// <param name="errorMessage">The error message describing the validation failure.</param>
    /// <param name="suggestedFixes">Optional list of suggested fixes.</param>
    public static ValidationResult Invalid(string errorMessage, List<string>? suggestedFixes = null)
    {
        return new ValidationResult(false, errorMessage, suggestedFixes ?? new List<string>());
    }

    /// <summary>
    /// Private constructor for ValidationResult.
    /// </summary>
    private ValidationResult(bool isValid, string errorMessage, List<string> suggestedFixes)
    {
        IsValid = isValid;
        ErrorMessage = errorMessage;
        SuggestedFixes = suggestedFixes;
    }
}
