using RitterFramework.Core;

namespace MLFramework.Diagnostics;

/// <summary>
/// Diagnostic formatter for broadcasting operations.
/// Provides specific error messages and suggestions for broadcasting-related issues.
/// </summary>
public class BroadcastDiagnosticFormatter : IDiagnosticFormatter
{
    /// <summary>
    /// Gets the operation type supported by this formatter.
    /// </summary>
    public OperationType SupportedOperation => OperationType.Broadcast;

    /// <summary>
    /// Formats a broadcasting error message.
    /// </summary>
    public string FormatError(ValidationResult result, params long[][] inputShapes)
    {
        if (inputShapes.Length < 2)
        {
            return $"Broadcast: Insufficient input shapes provided. {result.ErrorMessage}";
        }

        var shape1 = $"[{string.Join(", ", inputShapes[0])}]";
        var shape2 = $"[{string.Join(", ", inputShapes[1])}]";

        var message = $"Broadcast: {shape1} → {shape2}\n";

        // Check for broadcasting compatibility
        var compatibility = CheckBroadcastCompatibility(inputShapes[0], inputShapes[1]);
        if (compatibility == BroadcastCompatibility.Compatible)
        {
            message += $"Valid: Shapes are compatible for broadcasting";
        }
        else if (compatibility == BroadcastCompatibility.IncompatibleDimension)
        {
            var problemDim = FindProblematicDimension(inputShapes[0], inputShapes[1]);
            if (problemDim.HasValue)
            {
                message += $"Problem: Cannot broadcast dimension {problemDim.Value} without compatible shape";
            }
            else
            {
                message += $"Problem: {result.ErrorMessage}";
            }
        }
        else
        {
            message += $"Problem: {result.ErrorMessage}";
        }

        return message;
    }

    /// <summary>
    /// Generates suggestions for fixing broadcasting issues.
    /// </summary>
    public List<string> GenerateSuggestions(ValidationResult result)
    {
        var suggestions = new List<string>
        {
            "Check broadcasting rules - dimensions must be equal or one must be 1",
            "Consider using explicit reshape or expand operations instead of relying on broadcasting",
            "Verify that smaller tensor can be broadcast to match larger tensor shape"
        };

        // Add specific suggestions based on the error message
        if (result.ErrorMessage != null)
        {
            var errorMsg = result.ErrorMessage.ToLower();

            if (errorMsg.Contains("dimension"))
            {
                suggestions.Add("Ensure problematic dimensions are either equal or one of them is 1");
            }

            if (errorMsg.Contains("batch"))
            {
                suggestions.Add("Batch sizes must match or one must be 1 for broadcasting");
            }
        }

        return suggestions;
    }

    /// <summary>
    /// Checks if two shapes are compatible for broadcasting.
    /// </summary>
    private BroadcastCompatibility CheckBroadcastCompatibility(long[] shape1, long[] shape2)
    {
        int maxLength = Math.Max(shape1.Length, shape2.Length);
        long[] paddedShape1 = PadShape(shape1, maxLength);
        long[] paddedShape2 = PadShape(shape2, maxLength);

        for (int i = 0; i < maxLength; i++)
        {
            if (paddedShape1[i] != paddedShape2[i] && paddedShape1[i] != 1 && paddedShape2[i] != 1)
            {
                return BroadcastCompatibility.IncompatibleDimension;
            }
        }

        return BroadcastCompatibility.Compatible;
    }

    /// <summary>
    /// Finds the problematic dimension when broadcasting is not possible.
    /// </summary>
    private int? FindProblematicDimension(long[] shape1, long[] shape2)
    {
        int maxLength = Math.Max(shape1.Length, shape2.Length);
        long[] paddedShape1 = PadShape(shape1, maxLength);
        long[] paddedShape2 = PadShape(shape2, maxLength);

        for (int i = 0; i < maxLength; i++)
        {
            if (paddedShape1[i] != paddedShape2[i] && paddedShape1[i] != 1 && paddedShape2[i] != 1)
            {
                return i;
            }
        }

        return null;
    }

    /// <summary>
    /// Pads a shape array with leading 1s to match the desired length.
    /// </summary>
    private long[] PadShape(long[] shape, int targetLength)
    {
        if (shape.Length >= targetLength)
        {
            return shape;
        }

        var padded = new long[targetLength];
        int offset = targetLength - shape.Length;
        for (int i = 0; i < shape.Length; i++)
        {
            padded[i + offset] = shape[i];
        }

        return padded;
    }

    /// <summary>
    /// Enum for broadcast compatibility status.
    /// </summary>
    private enum BroadcastCompatibility
    {
        Compatible,
        IncompatibleDimension
    }
}
