namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Interface for operation-specific diagnostic handlers that provide specialized
/// validation, error messages, and suggestions for different operation types.
/// </summary>
public interface IOperationDiagnosticsHandler
{
    /// <summary>
    /// Validate shapes with operation-specific logic.
    /// </summary>
    /// <param name="inputShapes">Input tensor shapes.</param>
    /// <param name="operationParameters">Operation-specific parameters.</param>
    /// <returns>Validation result indicating success or failure with details.</returns>
    ValidationResult Validate(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters);

    /// <summary>
    /// Generate operation-specific error messages.
    /// </summary>
    /// <param name="inputShapes">Input tensor shapes.</param>
    /// <param name="operationParameters">Operation-specific parameters.</param>
    /// <param name="layerName">Name of the layer where the error occurred.</param>
    /// <returns>A formatted error message string.</returns>
    string GenerateErrorMessage(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters,
        string layerName);

    /// <summary>
    /// Generate operation-specific suggestions for fixing the issue.
    /// </summary>
    /// <param name="inputShapes">Input tensor shapes.</param>
    /// <param name="operationParameters">Operation-specific parameters.</param>
    /// <returns>List of suggested fixes.</returns>
    List<string> GenerateSuggestions(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters);

    /// <summary>
    /// Detect common patterns and potential issues in the operation.
    /// </summary>
    /// <param name="inputShapes">Input tensor shapes.</param>
    /// <param name="operationParameters">Operation-specific parameters.</param>
    /// <returns>List of detected issues or warnings.</returns>
    List<string> DetectIssues(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters);
}
