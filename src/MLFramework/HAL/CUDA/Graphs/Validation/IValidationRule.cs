namespace MLFramework.HAL.CUDA.Graphs.Validation;

/// <summary>
/// Interface for validation rules that check CUDA graph compatibility
/// </summary>
public interface IValidationRule
{
    /// <summary>
    /// Validates the graph and returns any errors or warnings
    /// </summary>
    /// <param name="graph">The graph to validate</param>
    /// <returns>Validation result with errors and warnings</returns>
    ValidationResult Validate(ICUDAGraph graph);

    /// <summary>
    /// Gets the name of this validation rule
    /// </summary>
    string RuleName { get; }

    /// <summary>
    /// Gets the description of what this rule validates
    /// </summary>
    string Description { get; }
}

/// <summary>
/// Result from a validation rule check
/// </summary>
public class ValidationResult
{
    /// <summary>
    /// Gets list of validation errors (empty if no errors)
    /// </summary>
    public List<string> Errors { get; }

    /// <summary>
    /// Gets list of warnings (non-critical issues)
    /// </summary>
    public List<string> Warnings { get; }

    public ValidationResult()
    {
        Errors = new List<string>();
        Warnings = new List<string>();
    }

    /// <summary>
    /// Checks if the validation passed (no errors)
    /// </summary>
    public bool IsValid => Errors.Count == 0;
}
