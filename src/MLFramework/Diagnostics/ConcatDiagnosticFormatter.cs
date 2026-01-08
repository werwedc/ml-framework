using RitterFramework.Core;

namespace MLFramework.Diagnostics;

/// <summary>
/// Diagnostic formatter for concatenation operations.
/// Provides specific error messages and suggestions for concatenation-related issues.
/// </summary>
public class ConcatDiagnosticFormatter : IDiagnosticFormatter
{
    /// <summary>
    /// Gets the operation type supported by this formatter.
    /// </summary>
    public OperationType SupportedOperation => OperationType.Concat;

    /// <summary>
    /// Formats a concatenation error message.
    /// </summary>
    public string FormatError(ValidationResult result, params long[][] inputShapes)
    {
        if (inputShapes.Length < 2)
        {
            return $"Concat: Insufficient input shapes provided. {result.ErrorMessage}";
        }

        var shape1 = $"[{string.Join(", ", inputShapes[0])}]";
        var shape2 = $"[{string.Join(", ", inputShapes[1])}]";

        var message = $"Concat: {shape1} + {shape2}\n";

        // Check if shapes are compatible for concatenation
        if (AreShapesCompatible(inputShapes[0], inputShapes[1], out int compatibleAxis))
        {
            message += $"Valid: All dimensions match except on axis {compatibleAxis}\n";

            // Calculate output shape
            var outputShape = (long[])inputShapes[0].Clone();
            outputShape[compatibleAxis] += inputShapes[1][compatibleAxis];
            message += $"Output shape: [{string.Join(", ", outputShape)}]";
        }
        else
        {
            message += $"Problem: {result.ErrorMessage}";
        }

        return message;
    }

    /// <summary>
    /// Generates suggestions for fixing concatenation issues.
    /// </summary>
    public List<string> GenerateSuggestions(ValidationResult result)
    {
        var suggestions = new List<string>
        {
            "Check tensor shapes - all dimensions must match except the concatenation axis",
            "Consider reshaping tensors to compatible dimensions before concatenation",
            "Verify the correct concatenation axis is specified"
        };

        // Add specific suggestions based on the error message
        if (result.ErrorMessage != null)
        {
            var errorMsg = result.ErrorMessage.ToLower();

            if (errorMsg.Contains("axis"))
            {
                suggestions.Add("Try concatenating on a different axis or reshape inputs to match dimensions");
            }

            if (errorMsg.Contains("dimension"))
            {
                suggestions.Add("Ensure batch sizes and other non-concatenation dimensions are identical");
            }
        }

        return suggestions;
    }

    /// <summary>
    /// Checks if two shapes are compatible for concatenation.
    /// </summary>
    private bool AreShapesCompatible(long[] shape1, long[] shape2, out int compatibleAxis)
    {
        compatibleAxis = -1;

        if (shape1.Length != shape2.Length)
        {
            return false;
        }

        int mismatchCount = 0;
        for (int i = 0; i < shape1.Length; i++)
        {
            if (shape1[i] != shape2[i])
            {
                mismatchCount++;
                compatibleAxis = i;
            }

            if (mismatchCount > 1)
            {
                return false;
            }
        }

        return mismatchCount == 1;
    }
}
