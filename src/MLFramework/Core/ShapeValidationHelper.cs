using RitterFramework.Core;
using MLFramework.Shapes;

namespace MLFramework.Core;

/// <summary>
/// Helper class for shape validation operations and exception creation.
/// </summary>
public static class ShapeValidationHelper
{
    /// <summary>
    /// Creates a ShapeMismatchException from a ValidationResult.
    /// </summary>
    /// <param name="result">The validation result.</param>
    /// <param name="operationType">The type of operation that failed.</param>
    /// <param name="layerName">The name of the layer/module (optional).</param>
    /// <param name="inputShapes">The input shapes that caused the failure.</param>
    /// <returns>A ShapeMismatchException with detailed information.</returns>
    public static MLFramework.Shapes.ShapeMismatchException CreateException(
        ValidationResult result,
        OperationType operationType,
        string? layerName = null,
        params long[][] inputShapes)
    {
        if (!result.IsValid)
        {
            var operationName = layerName != null ? $"{layerName}.{operationType}" : operationType.ToString();
            var problemDescription = BuildProblemDescription(operationType, inputShapes);
            var details = result.ErrorMessage;

            if (result.SuggestedFixes != null && result.SuggestedFixes.Count > 0)
            {
                details += Environment.NewLine + Environment.NewLine + "Suggested fixes:" + Environment.NewLine;
                for (int i = 0; i < result.SuggestedFixes.Count; i++)
                {
                    details += $"{i + 1}. {result.SuggestedFixes[i]}" + Environment.NewLine;
                }
            }

            var expectedShapes = inputShapes.Select(s => new SymbolicShape(s.ToList())).ToList();
            var actualShapes = inputShapes.Select(s => new SymbolicShape(s.ToList())).ToList();

            return new MLFramework.Shapes.ShapeMismatchException(
                operationName,
                expectedShapes,
                actualShapes,
                problemDescription + Environment.NewLine + details);
        }

        throw new ArgumentException("Cannot create exception from a valid ValidationResult", nameof(result));
    }

    /// <summary>
    /// Extracts a problem description from operation type and input shapes.
    /// </summary>
    /// <param name="operationType">The type of operation.</param>
    /// <param name="inputShapes">The input shapes.</param>
    /// <returns>A human-readable problem description.</returns>
    public static string BuildProblemDescription(OperationType operationType, params long[][] inputShapes)
    {
        if (inputShapes == null || inputShapes.Length == 0)
            return $"Shape validation failed for {operationType}.";

        return operationType switch
        {
            OperationType.MatrixMultiply or OperationType.Linear when inputShapes.Length >= 2 =>
                BuildMatrixMultiplyProblemDescription(inputShapes[0], inputShapes[1]),
            OperationType.Conv2D when inputShapes.Length >= 2 =>
                BuildConv2DProblemDescription(inputShapes[0], inputShapes[1]),
            OperationType.Concat =>
                BuildConcatProblemDescription(inputShapes),
            OperationType.Broadcast when inputShapes.Length >= 2 =>
                BuildBroadcastProblemDescription(inputShapes[0], inputShapes[1]),
            _ => $"Shape validation failed for {operationType} with {inputShapes.Length} input(s)."
        };
    }

    private static string BuildMatrixMultiplyProblemDescription(long[] shape1, long[] shape2)
    {
        return $"Matrix multiplication: Shape [{string.Join(", ", shape1)}] × [{string.Join(", ", shape2)}]";
    }

    private static string BuildConv2DProblemDescription(long[] inputShape, long[] kernelShape)
    {
        return $"Conv2D: Input [{string.Join(", ", inputShape)}] with kernel [{string.Join(", ", kernelShape)}]";
    }

    private static string BuildConcatProblemDescription(long[][] inputShapes)
    {
        var shapeStrings = inputShapes.Select(s => $"[{string.Join(", ", s)}]");
        return $"Concatenation of {inputShapes.Length} tensors: {string.Join(" + ", shapeStrings)}";
    }

    private static string BuildBroadcastProblemDescription(long[] shape1, long[] shape2)
    {
        return $"Broadcast: [{string.Join(", ", shape1)}] and [{string.Join(", ", shape2)}]";
    }
}
