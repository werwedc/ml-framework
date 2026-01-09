using System.Text;

namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Diagnostic handler for concatenation operations.
/// Provides specialized validation and error messages for concat.
/// </summary>
public class ConcatDiagnosticsHandler : BaseOperationDiagnosticsHandler
{
    /// <inheritdoc/>
    public override ValidationResult Validate(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var shapes = inputShapes.ToArray();

        if (shapes.Length < 2)
        {
            return ValidationResult.Failure("Concat requires at least 2 input tensors");
        }

        // Get concatenation axis (default: 0)
        int axis = operationParameters?.TryGetValue("axis", out var a) == true ? (int)a : 0;

        // All tensors should have same number of dimensions
        int dimCount = shapes[0].Length;
        for (int i = 1; i < shapes.Length; i++)
        {
            if (shapes[i].Length != dimCount)
            {
                return ValidationResult.Failure(
                    $"Tensor {i} has {shapes[i].Length} dimensions, but tensor 0 has {dimCount}");
            }
        }

        // All dimensions except axis should match
        for (int i = 0; i < shapes.Length; i++)
        {
            for (int d = 0; d < dimCount; d++)
            {
                if (d == axis) continue; // Skip concatenation axis

                if (shapes[i][d] != shapes[0][d])
                {
                    return ValidationResult.Failure(
                        $"Tensor {i} dimension {d} is {shapes[i][d]}, but tensor 0 has {shapes[0][d]}");
                }
            }
        }

        // Validate axis is within range
        if (axis < 0 || axis >= dimCount)
        {
            return ValidationResult.Failure(
                $"Axis {axis} is out of range for {dimCount}D tensors (valid range: 0 to {dimCount - 1})");
        }

        return ValidationResult.Success();
    }

    /// <inheritdoc/>
    public override string GenerateErrorMessage(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters,
        string layerName)
    {
        var shapes = inputShapes.ToArray();
        int axis = operationParameters?.TryGetValue("axis", out var a) == true ? (int)a : 0;

        var sb = new StringBuilder();
        sb.AppendLine($"Concat failed in layer '{layerName}' (axis={axis})");
        sb.AppendLine();

        sb.AppendLine("Input Shapes:");
        for (int i = 0; i < shapes.Length; i++)
        {
            sb.AppendLine($"  Tensor {i}: [{string.Join(", ", shapes[i])}]");
        }

        sb.AppendLine();
        sb.AppendLine("Expected: All tensors must have matching dimensions except at concatenation axis");

        // Find mismatching dimensions
        int dimCount = shapes[0].Length;
        for (int d = 0; d < dimCount; d++)
        {
            if (d == axis) continue;

            bool allMatch = true;
            long expectedDim = shapes[0][d];

            for (int i = 1; i < shapes.Length; i++)
            {
                if (shapes[i][d] != expectedDim)
                {
                    allMatch = false;
                    break;
                }
            }

            if (!allMatch)
            {
                sb.AppendLine();
                sb.AppendLine($"Problem: Dimension {d} does not match across all tensors");
                for (int i = 0; i < shapes.Length; i++)
                {
                    sb.AppendLine($"  Tensor {i}: {shapes[i][d]}");
                }
            }
        }

        return sb.ToString().Trim();
    }

    /// <inheritdoc/>
    public override List<string> GenerateSuggestions(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var suggestions = new List<string>();

        suggestions.Add("Ensure all tensors have the same number of dimensions");
        suggestions.Add("Check that all non-concatenation dimensions match exactly");
        suggestions.Add("Consider using unsqueeze/squeeze to add/remove dimensions before concatenation");
        suggestions.Add("Verify the concatenation axis is correct");

        // Detect axis issues
        int axis = operationParameters?.TryGetValue("axis", out var a) == true ? (int)a : 0;
        var shapes = inputShapes.ToArray();

        int dimCount = shapes[0].Length;
        if (axis < 0 || axis >= dimCount)
        {
            suggestions.Add($"Axis {axis} is invalid - try axis 0, 1, 2, or 3");
        }

        // Check for transpose issues
        bool possibleTranspose = true;
        for (int d = 0; d < dimCount; d++)
        {
            long firstDim = shapes[0][d];
            for (int i = 1; i < shapes.Length; i++)
            {
                if (shapes[i][d] != firstDim)
                {
                    possibleTranspose = false;
                    break;
                }
            }
        }

        if (possibleTranspose && shapes.Length == 2)
        {
            suggestions.Add("All dimensions match - did you mean to use stack instead of concat?");
        }

        return suggestions;
    }
}
