namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Provides comprehensive formatting methods for generating human-readable diagnostic reports
/// from shape mismatch exceptions.
/// </summary>
public static class ShapeReportFormatter
{
    /// <summary>
    /// Formats a shape mismatch exception into a detailed diagnostic report.
    /// </summary>
    /// <param name="exception">The exception to format.</param>
    /// <returns>A formatted multi-line diagnostic report.</returns>
    public static string Format(ShapeMismatchException exception)
    {
        var sb = new System.Text.StringBuilder();

        // Header
        sb.AppendLine("=".PadRight(80, '='));
        sb.AppendLine("SHAPE MISMATCH DIAGNOSTIC REPORT");
        sb.AppendLine("=".PadRight(80, '='));
        sb.AppendLine();

        // Summary
        sb.AppendLine("Summary:");
        sb.AppendLine($"  Layer: {exception.LayerName}");
        sb.AppendLine($"  Operation: {exception.OperationType}");
        sb.AppendLine();

        // Input Shapes
        if (exception.InputShapes != null && exception.InputShapes.Count > 0)
        {
            sb.AppendLine("Input Shapes:");
            for (int i = 0; i < exception.InputShapes.Count; i++)
            {
                var shape = exception.InputShapes[i];
                var shapeStr = FormatShape(shape);
                var label = GetShapeLabel(exception.OperationType, i);
                sb.AppendLine($"  {label}: {shapeStr}");
            }
            sb.AppendLine();
        }

        // Expected Shapes
        if (exception.ExpectedShapes != null && exception.ExpectedShapes.Count > 0)
        {
            sb.AppendLine("Expected Shapes:");
            for (int i = 0; i < exception.ExpectedShapes.Count; i++)
            {
                var shape = exception.ExpectedShapes[i];
                sb.AppendLine($"  Expected {i}: {FormatShape(shape)}");
            }
            sb.AppendLine();
        }

        // Problem Description
        sb.AppendLine("Problem:");
        sb.Append("  ");
        sb.AppendLine(exception.ProblemDescription ?? "No problem description provided.");
        sb.AppendLine();

        // Context
        if (!string.IsNullOrEmpty(exception.LayerName))
        {
            sb.AppendLine("Context:");
            sb.AppendLine($"  Current Layer: {exception.LayerName} ({exception.OperationType})");
            sb.AppendLine();
        }

        // Suggested Fixes
        if (exception.SuggestedFixes != null && exception.SuggestedFixes.Count > 0)
        {
            sb.AppendLine("Suggested Fixes:");
            for (int i = 0; i < exception.SuggestedFixes.Count; i++)
            {
                sb.AppendLine($"  {i + 1}. {exception.SuggestedFixes[i]}");
            }
            sb.AppendLine();
        }

        // Footer
        sb.AppendLine("=".PadRight(80, '='));
        sb.AppendLine($"Generated at: {System.DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
        sb.AppendLine("=".PadRight(80, '='));

        return sb.ToString();
    }

    /// <summary>
    /// Formats a detailed report with additional visualization and dimension analysis.
    /// </summary>
    /// <param name="exception">The exception to format.</param>
    /// <returns>A detailed report with visualizations.</returns>
    public static string FormatDetailed(ShapeMismatchException exception)
    {
        // Include additional details like stack trace suggestions,
        // visual ASCII art of shapes, etc.
        var baseReport = Format(exception);
        var detailed = new System.Text.StringBuilder(baseReport);

        detailed.AppendLine();
        detailed.AppendLine("Additional Details:");
        detailed.AppendLine();

        // Add shape visualizations
        if (exception.InputShapes != null && exception.InputShapes.Count > 0)
        {
            detailed.AppendLine("Shape Visualization:");
            for (int i = 0; i < exception.InputShapes.Count; i++)
            {
                var shape = exception.InputShapes[i];
                detailed.AppendLine($"  Input {i}:");
                detailed.AppendLine(VisualizeShape(shape));
                detailed.AppendLine();
            }
        }

        // Add dimension analysis
        detailed.AppendLine("Dimension Analysis:");
        detailed.AppendLine(AnalyzeDimensions(exception));
        detailed.AppendLine();

        return detailed.ToString();
    }

    /// <summary>
    /// Formats a one-line summary of the exception.
    /// </summary>
    /// <param name="exception">The exception to format.</param>
    /// <returns>A one-line summary string.</returns>
    public static string FormatSummary(ShapeMismatchException exception)
    {
        return $"Shape mismatch in '{exception.LayerName}' ({exception.OperationType}): " +
               $"{exception.ProblemDescription}";
    }

    /// <summary>
    /// Formats a shape array into a readable string representation.
    /// </summary>
    /// <param name="shape">The shape array to format.</param>
    /// <returns>A formatted shape string.</returns>
    private static string FormatShape(long[]? shape)
    {
        if (shape == null) return "null";

        var formatted = shape.Select(d => d == -1 ? "?" : d.ToString());
        return $"[{string.Join(", ", formatted)}]";
    }

    /// <summary>
    /// Gets a descriptive label for a shape based on the operation type and index.
    /// </summary>
    /// <param name="operation">The operation type.</param>
    /// <param name="index">The shape index.</param>
    /// <returns>A descriptive label.</returns>
    private static string GetShapeLabel(OperationType operation, int index)
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

    /// <summary>
    /// Creates a simple ASCII visualization of a tensor shape.
    /// </summary>
    /// <param name="shape">The shape to visualize.</param>
    /// <returns>An ASCII art representation of the shape.</returns>
    private static string VisualizeShape(long[]? shape)
    {
        if (shape == null || shape.Length == 0)
            return "  (empty)";

        var sb = new System.Text.StringBuilder();
        sb.Append("  ");

        // Simple ASCII visualization
        for (int i = 0; i < shape.Length; i++)
        {
            sb.Append($"[{shape[i]}]");
            if (i < shape.Length - 1)
            {
                sb.Append(" → ");
            }
        }

        // Add dimension labels
        sb.AppendLine();
        sb.Append("  ");
        for (int i = 0; i < shape.Length; i++)
        {
            var label = GetDimensionLabel(i, shape.Length);
            var padding = System.Math.Max(shape[i].ToString().Length, 2);
            sb.Append($" {label} ".PadLeft(padding + 2));
            if (i < shape.Length - 1)
            {
                sb.Append("    ");
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Gets a label for a specific dimension based on its position and total dimensions.
    /// </summary>
    /// <param name="index">The dimension index.</param>
    /// <param name="totalDimensions">The total number of dimensions.</param>
    /// <returns>A dimension label.</returns>
    private static string GetDimensionLabel(int index, int totalDimensions)
    {
        switch (totalDimensions)
        {
            case 1:
                return "";
            case 2:
                return index == 0 ? "rows" : "cols";
            case 4:
                return index switch
                {
                    0 => "batch",
                    1 => "ch",
                    2 => "h",
                    3 => "w",
                    _ => ""
                };
            default:
                return $"d{index}";
        }
    }

    /// <summary>
    /// Analyzes dimension mismatches between input shapes.
    /// </summary>
    /// <param name="exception">The exception containing shape information.</param>
    /// <returns>A string describing dimension analysis.</returns>
    private static string AnalyzeDimensions(ShapeMismatchException exception)
    {
        var analysis = new System.Text.StringBuilder();

        if (exception.InputShapes == null || exception.InputShapes.Count == 0)
            return "  No shape information available";

        // Analyze dimension mismatches
        for (int i = 0; i < exception.InputShapes.Count - 1; i++)
        {
            var shape1 = exception.InputShapes[i];
            var shape2 = exception.InputShapes[i + 1];

            if (shape1 == null || shape2 == null)
                continue;

            var minDim = System.Math.Min(shape1.Length, shape2.Length);

            for (int d = 0; d < minDim; d++)
            {
                if (shape1[d] != shape2[d])
                {
                    analysis.AppendLine($"  Dimension {d}: Tensor {i} has {shape1[d]}, Tensor {i + 1} has {shape2[d]}");
                    analysis.AppendLine($"    → Mismatch: {System.Math.Abs(shape1[d] - shape2[d])} difference");

                    if (shape1[d] != 0 && shape2[d] != 0 &&
                        (shape1[d] % shape2[d] == 0 || shape2[d] % shape1[d] == 0))
                    {
                        analysis.AppendLine($"    → These dimensions are multiples (possibly a broadcast issue)");
                    }
                }
            }

            // Check for different dimension counts
            if (shape1.Length != shape2.Length)
            {
                analysis.AppendLine($"  Dimension count mismatch: Tensor {i} has {shape1.Length}D, Tensor {i + 1} has {shape2.Length}D");
            }
        }

        if (analysis.Length == 0)
        {
            analysis.AppendLine("  No obvious dimension mismatches found (error may be semantic)");
        }

        return analysis.ToString();
    }
}
