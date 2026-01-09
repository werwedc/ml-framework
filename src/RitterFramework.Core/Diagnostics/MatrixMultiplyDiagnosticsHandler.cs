using System.Text;

namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Diagnostic handler for matrix multiplication operations.
/// Provides specialized validation and error messages for matrix multiplication.
/// </summary>
public class MatrixMultiplyDiagnosticsHandler : BaseOperationDiagnosticsHandler
{
    /// <inheritdoc/>
    public override ValidationResult Validate(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var shapes = inputShapes.ToArray();

        if (shapes.Length != 2)
        {
            return ValidationResult.Failure("Matrix multiplication requires exactly 2 input tensors");
        }

        var shapeA = shapes[0];
        var shapeB = shapes[1];

        // Handle different dimensionality
        if (shapeA.Length < 2 || shapeB.Length < 2)
        {
            return ValidationResult.Failure("Both tensors must have at least 2 dimensions");
        }

        // Get inner dimensions
        int innerDimA = shapeA.Length - 1;
        int innerDimB = shapeB.Length - 2;

        if (shapeA[innerDimA] != shapeB[innerDimB])
        {
            return ValidationResult.Failure(
                $"Inner dimensions mismatch: {shapeA[innerDimA]} != {shapeB[innerDimB]}");
        }

        // Check for potential batch dimension mismatch
        if (shapeA.Length == 3 && shapeB.Length == 2)
        {
            // shapeA: [batch, m, k], shapeB: [k, n] - this is fine
        }
        else if (shapeA.Length == 2 && shapeB.Length == 3)
        {
            // shapeA: [m, k], shapeB: [batch, k, n] - check if k matches
            if (shapeA[1] != shapeB[1])
            {
                return ValidationResult.Failure(
                    $"Batch dimension mismatch: {shapeA[1]} != {shapeB[1]}");
            }
        }
        else if (shapeA.Length == shapeB.Length && shapeA.Length > 2)
        {
            // Check batch dimensions match
            for (int i = 0; i < shapeA.Length - 2; i++)
            {
                if (shapeA[i] != shapeB[i])
                {
                    return ValidationResult.Failure(
                        $"Batch dimension {i} mismatch: {shapeA[i]} != {shapeB[i]}");
                }
            }
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
        var shapeA = shapes[0];
        var shapeB = shapes[1];

        var sb = new StringBuilder();
        sb.AppendLine($"Matrix multiplication failed in layer '{layerName}'");
        sb.AppendLine();

        if (shapeA.Length == 2 && shapeB.Length == 2)
        {
            sb.AppendLine($"Input shape:    [{shapeA[0]}, {shapeA[1]}]");
            sb.AppendLine($"Weight shape:   [{shapeB[0]}, {shapeB[1]}]");
            sb.AppendLine();
            sb.AppendLine($"Expected:       [m, k] × [k, n] → [m, n]");
            sb.AppendLine($"                → Requires k to match");
            sb.AppendLine();
            sb.AppendLine($"Problem: Dimension 1 of input ({shapeA[1]}) does not match dimension 0 of weight ({shapeB[0]})");
        }
        else if (shapeA.Length == 3)
        {
            sb.AppendLine($"Input shape:    [{shapeA[0]}, {shapeA[1]}, {shapeA[2]}]");
            sb.AppendLine($"Weight shape:   [{shapeB[0]}, {shapeB[1]}]");
            sb.AppendLine();
            sb.AppendLine($"Expected:       [batch, m, k] × [k, n] → [batch, m, n]");
            sb.AppendLine($"                → Requires k to match");
            sb.AppendLine();
            sb.AppendLine($"Problem: Dimension 2 of input ({shapeA[2]}) does not match dimension 0 of weight ({shapeB[0]})");
        }

        return sb.ToString().Trim();
    }

    /// <inheritdoc/>
    public override List<string> GenerateSuggestions(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var suggestions = new List<string>();
        var shapes = inputShapes.ToArray();
        var shapeA = shapes[0];
        var shapeB = shapes[1];

        // Common Matrix Multiply issues and fixes
        if (shapeA.Length == 2 && shapeB.Length == 2)
        {
            suggestions.Add($"Transpose weight matrix if needed: currently [{shapeB[0]}, {shapeB[1]}] → [{shapeB[1]}, {shapeB[0]}]");
            suggestions.Add($"Check input tensor is in correct format (not transposed)");
        }
        else if (shapeA.Length == 3)
        {
            suggestions.Add($"Ensure weight matrix shape matches last dimension of input");
            suggestions.Add($"Check if weight matrix should be transposed");
        }

        suggestions.Add("Verify model configuration matches expected architecture");
        suggestions.Add("Consider using torch.nn.Linear with correct in_features parameter");

        return suggestions;
    }

    /// <inheritdoc/>
    public override List<string> DetectIssues(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        var issues = new List<string>();
        var shapes = inputShapes.ToArray();
        var shapeA = shapes[0];
        var shapeB = shapes[1];

        // Check for transpose issue (common mistake)
        if (shapeA.Length == 2 && shapeB.Length == 2)
        {
            if (shapeA[1] == shapeB[1])
            {
                issues.Add("Possible transpose issue: input and weight have matching last dimension");
            }
        }

        // Check for scalar multiplication
        if (shapeA.Length == 1 || shapeB.Length == 1)
        {
            issues.Add("One of the tensors is 1D - use squeeze/unsqueeze or proper broadcasting");
        }

        return issues;
    }
}
