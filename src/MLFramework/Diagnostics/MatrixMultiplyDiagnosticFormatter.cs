using RitterFramework.Core;

namespace MLFramework.Diagnostics;

/// <summary>
/// Diagnostic formatter for matrix multiplication operations.
/// Provides specific error messages and suggestions for shape mismatch issues.
/// </summary>
public class MatrixMultiplyDiagnosticFormatter : IDiagnosticFormatter
{
    /// <summary>
    /// Gets the operation type supported by this formatter.
    /// </summary>
    public OperationType SupportedOperation => OperationType.MatrixMultiply;

    /// <summary>
    /// Formats a matrix multiplication error message.
    /// </summary>
    public string FormatError(ValidationResult result, params long[][] inputShapes)
    {
        if (inputShapes.Length < 2)
        {
            return $"Matrix multiplication: Insufficient input shapes provided. {result.ErrorMessage}";
        }

        var shape1 = $"[{string.Join(", ", inputShapes[0])}]";
        var shape2 = $"[{string.Join(", ", inputShapes[1])}]";

        var message = $"Matrix multiplication: Shape {shape1} × {shape2} invalid\n";

        // Extract inner dimensions
        if (inputShapes[0].Length >= 2 && inputShapes[1].Length >= 2)
        {
            var innerDim1 = inputShapes[0][inputShapes[0].Length - 1];
            var innerDim2 = inputShapes[1][inputShapes[1].Length - 2];

            message += $"Problem: Inner dimensions {innerDim1} and {innerDim2} must match";
        }
        else
        {
            message += $"Problem: {result.ErrorMessage}";
        }

        return message;
    }

    /// <summary>
    /// Generates suggestions for fixing matrix multiplication issues.
    /// </summary>
    public List<string> GenerateSuggestions(ValidationResult result)
    {
        var suggestions = new List<string>
        {
            "Check layer configurations - ensure output features of previous layer match input features of this layer",
            "Verify tensor shapes before matrix multiplication",
            "Consider transposing one of the matrices if dimensions are swapped"
        };

        // Add specific suggestions based on the error message
        if (result.ErrorMessage != null)
        {
            var errorMsg = result.ErrorMessage.ToLower();

            if (errorMsg.Contains("256") && errorMsg.Contains("128"))
            {
                suggestions.Add("Previous layer outputs 256 features, but this layer expects 128. Adjust layer configuration to match");
            }
        }

        return suggestions;
    }
}
