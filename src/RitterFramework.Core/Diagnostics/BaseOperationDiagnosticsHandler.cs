namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Abstract base class for operation-specific diagnostic handlers.
/// Provides common utility methods and default implementations.
/// </summary>
public abstract class BaseOperationDiagnosticsHandler : IOperationDiagnosticsHandler
{
    /// <inheritdoc/>
    public abstract ValidationResult Validate(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters);

    /// <inheritdoc/>
    public abstract string GenerateErrorMessage(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters,
        string layerName);

    /// <inheritdoc/>
    public virtual List<string> GenerateSuggestions(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        return new List<string>();
    }

    /// <inheritdoc/>
    public virtual List<string> DetectIssues(
        IEnumerable<long[]> inputShapes,
        IDictionary<string, object> operationParameters)
    {
        return new List<string>();
    }

    /// <summary>
    /// Check if two shapes are broadcast-compatible.
    /// </summary>
    /// <param name="shape1">First shape.</param>
    /// <param name="shape2">Second shape.</param>
    /// <returns>True if shapes can be broadcast together.</returns>
    protected bool CheckBroadcastCompatibility(long[] shape1, long[] shape2)
    {
        // Check if shapes are broadcast-compatible
        int dim1 = shape1.Length;
        int dim2 = shape2.Length;
        int maxDim = Math.Max(dim1, dim2);

        for (int i = 1; i <= maxDim; i++)
        {
            int idx1 = dim1 - i;
            int idx2 = dim2 - i;

            long dim1Val = idx1 >= 0 ? shape1[idx1] : 1;
            long dim2Val = idx2 >= 0 ? shape2[idx2] : 1;

            if (dim1Val != dim2Val && dim1Val != 1 && dim2Val != 1)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Format a shape array as a string, using '?' for unknown dimensions (-1).
    /// </summary>
    protected string FormatShape(long[] shape)
    {
        if (shape == null) return "null";

        var formatted = shape.Select(d => d == -1 ? "?" : d.ToString());
        return $"[{string.Join(", ", formatted)}]";
    }

    /// <summary>
    /// Get a human-readable label for an input tensor based on operation type.
    /// </summary>
    protected string GetShapeLabel(OperationType operation, int index)
    {
        switch (operation)
        {
            case OperationType.MatrixMultiply:
                return index == 0 ? "Input" : "Weight";
            case OperationType.Conv2D:
            case OperationType.Conv1D:
                return index == 0 ? "Input" : "Kernel";
            case OperationType.Concat:
            case OperationType.Stack:
                return $"Input {index}";
            default:
                return $"Tensor {index}";
        }
    }
}
