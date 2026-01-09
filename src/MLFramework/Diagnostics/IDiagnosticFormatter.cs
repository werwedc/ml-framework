using RitterFramework.Core;

namespace MLFramework.Diagnostics;

/// <summary>
/// Interface for operation-specific diagnostic formatters that generate
/// human-readable error messages and suggestions for different operation types.
/// </summary>
public interface IDiagnosticFormatter
{
    /// <summary>
    /// Gets the operation type this formatter supports.
    /// </summary>
    OperationType SupportedOperation { get; }

    /// <summary>
    /// Formats a validation error message with context about the input shapes.
    /// </summary>
    /// <param name="result">The validation result containing error information.</param>
    /// <param name="inputShapes">The shapes of the input tensors that caused the error.</param>
    /// <returns>A formatted human-readable error message.</returns>
    string FormatError(ValidationResult result, params long[][] inputShapes);

    /// <summary>
    /// Generates a list of actionable suggestions for fixing the validation issue.
    /// </summary>
    /// <param name="result">The validation result containing error information.</param>
    /// <returns>A list of suggested fixes.</returns>
    List<string> GenerateSuggestions(ValidationResult result);
}
