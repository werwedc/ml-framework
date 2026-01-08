using RitterFramework.Core.Diagnostics;

namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Generates actionable fix suggestions for common shape mismatch scenarios using pattern detection.
/// </summary>
public class FixSuggestionGenerator
{
    /// <summary>
    /// Generates fix suggestions based on operation type and shape information.
    /// </summary>
    /// <param name="operationType">The type of operation being performed.</param>
    /// <param name="inputShapes">The shapes of the input tensors.</param>
    /// <param name="expectedShapes">The expected shapes for the operation.</param>
    /// <returns>A list of actionable fix suggestions.</returns>
    public List<string> GenerateSuggestions(
        OperationType operationType,
        long[][] inputShapes,
        long[][] expectedShapes)
    {
        if (inputShapes == null || inputShapes.Length == 0)
        {
            return new List<string>();
        }

        var suggestions = new List<string>();

        // Pattern detection based on operation type
        switch (operationType)
        {
            case OperationType.Linear:
                suggestions.AddRange(DetectLinearPatterns(inputShapes, expectedShapes));
                break;
            case OperationType.Conv2D:
                suggestions.AddRange(DetectConv2DPatterns(inputShapes, expectedShapes));
                break;
            case OperationType.Concat:
                suggestions.AddRange(DetectConcatPatterns(inputShapes));
                break;
            case OperationType.Broadcast:
                suggestions.AddRange(DetectBroadcastPatterns(inputShapes));
                break;
            case OperationType.MatrixMultiply:
                suggestions.AddRange(DetectMatMulPatterns(inputShapes, expectedShapes));
                break;
            default:
                suggestions.AddRange(DetectGenericPatterns(inputShapes, expectedShapes));
                break;
        }

        // Add generic patterns that apply to all operations
        suggestions.AddRange(DetectCommonPatterns(inputShapes, expectedShapes));

        return suggestions;
    }

    /// <summary>
    /// Detects patterns specific to linear layer operations.
    /// </summary>
    private List<string> DetectLinearPatterns(long[][] inputShapes, long[][] expectedShapes)
    {
        var suggestions = new List<string>();

        if (inputShapes.Length == 0 || expectedShapes == null || expectedShapes.Length == 0)
        {
            return suggestions;
        }

        var inputShape = inputShapes[0];
        var expectedShape = expectedShapes[0];

        // Pattern 3: Feature Size Mismatch in Linear Layer
        // Input: [32, 256], Expected: [32, 128]
        if (inputShape.Length >= 2 && expectedShape.Length >= 2)
        {
            long actualFeatures = inputShape[^1]; // Last dimension
            long expectedFeatures = expectedShape[^1];

            if (actualFeatures != expectedFeatures)
            {
                suggestions.Add(string.Format(
                    SuggestionTemplates.FeatureSizeMismatch,
                    actualFeatures,
                    expectedFeatures));
            }
        }

        // Pattern 4: Transpose Required
        // Input: [10, 32], Weight: [10, 64]
        if (inputShape.Length >= 2 && inputShape[0] > 0 && inputShape[1] > 0)
        {
            if (inputShape[0] == inputShape[1])
            {
                suggestions.Add(string.Format(
                    SuggestionTemplates.TransposeRequired,
                    $"[{string.Join(", ", inputShape)}]",
                    $"[{string.Join(", ", inputShape.Reverse())}]"));
            }
        }

        return suggestions;
    }

    /// <summary>
    /// Detects patterns specific to convolution operations.
    /// </summary>
    private List<string> DetectConv2DPatterns(long[][] inputShapes, long[][] expectedShapes)
    {
        var suggestions = new List<string>();

        if (inputShapes.Length == 0 || expectedShapes == null || expectedShapes.Length == 0)
        {
            return suggestions;
        }

        var inputShape = inputShapes[0];
        var expectedShape = expectedShapes[0];

        // Pattern 2: Channel Order Mismatch
        // Input: [32, 224, 224, 3], Expected: [32, 3, 224, 224]
        if (inputShape.Length == 4 && expectedShape.Length == 4)
        {
            // Check if NHWC to NCHW conversion is needed
            if (IsChannelOrderMismatch(inputShape, expectedShape))
            {
                suggestions.Add(string.Format(
                    SuggestionTemplates.ChannelOrderMismatch,
                    "NHWC",
                    "NCHW"));
            }
        }

        return suggestions;
    }

    /// <summary>
    /// Detects patterns specific to concatenation operations.
    /// </summary>
    private List<string> DetectConcatPatterns(long[][] inputShapes)
    {
        var suggestions = new List<string>();

        if (inputShapes.Length < 2)
        {
            return suggestions;
        }

        var shape1 = inputShapes[0];
        var shape2 = inputShapes[1];

        // Pattern 5: Concatenation Dimension Mismatch
        // Input1: [32, 128], Input2: [32, 256], Axis: 0
        // Suggest axis 1 instead if axis 0 has different sizes
        for (int axis = 0; axis < Math.Min(shape1.Length, shape2.Length); axis++)
        {
            if (shape1[axis] != shape2[axis])
            {
                // Find a dimension that matches
                for (int alternativeAxis = 0; alternativeAxis < Math.Min(shape1.Length, shape2.Length); alternativeAxis++)
                {
                    if (alternativeAxis != axis && shape1[alternativeAxis] == shape2[alternativeAxis])
                    {
                        suggestions.Add(string.Format(
                            SuggestionTemplates.ConcatenationDimensionMismatch,
                            axis,
                            alternativeAxis));
                        break;
                    }
                }
                break;
            }
        }

        return suggestions;
    }

    /// <summary>
    /// Detects patterns specific to broadcasting operations.
    /// </summary>
    private List<string> DetectBroadcastPatterns(long[][] inputShapes)
    {
        var suggestions = new List<string>();

        if (inputShapes.Length < 2)
        {
            return suggestions;
        }

        var shape1 = inputShapes[0];
        var shape2 = inputShapes[1];

        // Pattern 6: Broadcasting Failure
        // Input1: [32, 10], Input2: [20, 10]
        if (shape1.Length == shape2.Length && shape1.Length > 0)
        {
            bool canBroadcast = true;

            for (int i = 0; i < shape1.Length; i++)
            {
                if (shape1[i] != shape2[i] && shape1[i] != 1 && shape2[i] != 1)
                {
                    canBroadcast = false;
                    break;
                }
            }

            if (!canBroadcast)
            {
                suggestions.Add(string.Format(
                    SuggestionTemplates.BroadcastingFailure,
                    $"[{string.Join(", ", shape1)}]",
                    $"[{string.Join(", ", shape2)}]"));
            }
        }

        return suggestions;
    }

    /// <summary>
    /// Detects patterns specific to matrix multiplication operations.
    /// </summary>
    private List<string> DetectMatMulPatterns(long[][] inputShapes, long[][] expectedShapes)
    {
        var suggestions = new List<string>();

        if (inputShapes.Length == 0 || expectedShapes == null || expectedShapes.Length == 0)
        {
            return suggestions;
        }

        var inputShape = inputShapes[0];
        var expectedShape = expectedShapes[0];

        // Check for transpose requirement in matrix multiplication
        if (inputShape.Length >= 2 && expectedShape.Length >= 2)
        {
            // For matrix multiply, if inner dimensions don't match, suggest transpose
            if (inputShape[^1] != expectedShape[^2])
            {
                suggestions.Add(string.Format(
                    SuggestionTemplates.TransposeRequired,
                    $"[{string.Join(", ", inputShape)}]",
                    $"[{string.Join(", ", inputShape.Reverse())}]"));
            }
        }

        return suggestions;
    }

    /// <summary>
    /// Detects generic patterns that don't fit specific operation types.
    /// </summary>
    private List<string> DetectGenericPatterns(long[][] inputShapes, long[][] expectedShapes)
    {
        var suggestions = new List<string>();

        if (inputShapes.Length == 0 || expectedShapes == null || expectedShapes.Length == 0)
        {
            return suggestions;
        }

        var inputShape = inputShapes[0];
        var expectedShape = expectedShapes[0];

        suggestions.Add(string.Format(
            SuggestionTemplates.GenericShapeMismatch,
            inputShape.Length == expectedShape.Length ? "operation" : "reshape",
            $"[{string.Join(", ", inputShape)}]",
            $"[{string.Join(", ", expectedShape)}]"));

        return suggestions;
    }

    /// <summary>
    /// Detects common patterns that apply to all operations.
    /// </summary>
    private List<string> DetectCommonPatterns(long[][] inputShapes, long[][] expectedShapes)
    {
        var suggestions = new List<string>();

        if (inputShapes.Length == 0)
        {
            return suggestions;
        }

        var inputShape = inputShapes[0];

        // Pattern 1: Missing Batch Dimension
        // Input: [784], Expected: [*, 784]
        if (expectedShapes != null && expectedShapes.Length > 0)
        {
            var expectedShape = expectedShapes[0];

            if (inputShape.Length + 1 == expectedShape.Length)
            {
                // Check if all dimensions match except for the leading one
                bool dimensionsMatch = true;
                for (int i = 0; i < inputShape.Length; i++)
                {
                    if (inputShape[i] != expectedShape[i + 1])
                    {
                        dimensionsMatch = false;
                        break;
                    }
                }

                if (dimensionsMatch)
                {
                    suggestions.Add(string.Format(
                        SuggestionTemplates.MissingBatchDim,
                        string.Join(", ", inputShape)));
                }
            }
        }

        // Check for squeeze opportunity
        if (inputShape.Contains(1))
        {
            for (int i = 0; i < inputShape.Length; i++)
            {
                if (inputShape[i] == 1)
                {
                    suggestions.Add(string.Format(
                        SuggestionTemplates.SqueezeSuggestion,
                        i));
                    break;
                }
            }
        }

        return suggestions;
    }

    /// <summary>
    /// Checks if two 4D shapes represent a channel order mismatch (NHWC vs NCHW).
    /// </summary>
    private bool IsChannelOrderMismatch(long[] shape1, long[] shape2)
    {
        if (shape1.Length != 4 || shape2.Length != 4)
        {
            return false;
        }

        // Check if shapes are NHWC and NCHW respectively
        // NHWC: [batch, height, width, channels]
        // NCHW: [batch, channels, height, width]

        bool shape1IsNHWC = shape1[1] == shape2[2] && shape1[2] == shape2[3] && shape1[3] == shape2[1];
        bool shape2IsNHWC = shape2[1] == shape1[2] && shape2[2] == shape1[3] && shape2[3] == shape1[1];

        return shape1IsNHWC || shape2IsNHWC;
    }
}
