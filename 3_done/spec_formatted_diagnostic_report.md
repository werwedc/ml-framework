# Technical Spec: Formatted Diagnostic Report

## Overview
Implement comprehensive formatting methods for generating human-readable diagnostic reports. This includes the `GetDiagnosticReport()` method on `ShapeMismatchException` and additional formatting utilities for logs, HTML reports, and JSON exports.

## Requirements

### Enhanced ShapeMismatchException.GetDiagnosticReport()
Update the existing `GetDiagnosticReport()` method to provide detailed, well-formatted output:

```csharp
public class ShapeMismatchException : Exception
{
    // ... existing properties ...

    public string GetDiagnosticReport()
    {
        return ShapeReportFormatter.Format(this);
    }

    // Generate multi-line formatted report
    public string GetDetailedReport()
    {
        return ShapeReportFormatter.FormatDetailed(this);
    }

    // Generate one-line summary
    public string GetSummary()
    {
        return ShapeReportFormatter.FormatSummary(this);
    }
}
```

### ShapeReportFormatter Class
```csharp
public static class ShapeReportFormatter
{
    public static string Format(ShapeMismatchException exception)
    {
        var sb = new StringBuilder();

        // Header
        sb.AppendLine("=".PadRight(80, '='));
        sb.AppendLine("SHAPE MISMATCH DIAGNOSTIC REPORT");
        sb.AppendLine("=".PadRight(80, '='));
        sb.AppendLine();

        // Summary
        sb.AppendLine("Summary:");
        sb.AppendLine($"  Layer: {exception.LayerName}");
        sb.AppendLine($"  Operation: {exception.OperationType}");
        if (exception.BatchSize.HasValue)
        {
            sb.AppendLine($"  Batch Size: {exception.BatchSize.Value}");
        }
        sb.AppendLine();

        // Input Shapes
        if (exception.InputShapes != null && exception.InputShapes.Any())
        {
            sb.AppendLine("Input Shapes:");
            for (int i = 0; i < exception.InputShapes.Count(); i++)
            {
                var shape = exception.InputShapes.ElementAt(i);
                var shapeStr = FormatShape(shape);
                var label = GetShapeLabel(exception.OperationType, i);
                sb.AppendLine($"  {label}: {shapeStr}");
            }
            sb.AppendLine();
        }

        // Expected Shapes
        if (exception.ExpectedShapes != null && exception.ExpectedShapes.Any())
        {
            sb.AppendLine("Expected Shapes:");
            for (int i = 0; i < exception.ExpectedShapes.Count(); i++)
            {
                var shape = exception.ExpectedShapes.ElementAt(i);
                sb.AppendLine($"  Expected {i}: {FormatShape(shape)}");
            }
            sb.AppendLine();
        }

        // Problem Description
        sb.AppendLine("Problem:");
        sb.Append("  ");
        sb.AppendLine(exception.ProblemDescription);
        sb.AppendLine();

        // Context
        if (!string.IsNullOrEmpty(exception.PreviousLayerContext) ||
            !string.IsNullOrEmpty(exception.LayerName))
        {
            sb.AppendLine("Context:");
            if (!string.IsNullOrEmpty(exception.LayerName))
            {
                sb.AppendLine($"  Current Layer: {exception.LayerName} ({exception.OperationType})");
            }
            if (!string.IsNullOrEmpty(exception.PreviousLayerContext))
            {
                sb.AppendLine($"  {exception.PreviousLayerContext}");
            }
            sb.AppendLine();
        }

        // Suggested Fixes
        if (exception.SuggestedFixes != null && exception.SuggestedFixes.Any())
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
        sb.AppendLine($"Generated at: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
        sb.AppendLine("=".PadRight(80, '='));

        return sb.ToString();
    }

    public static string FormatDetailed(ShapeMismatchException exception)
    {
        // Include additional details like stack trace suggestions,
        // visual ASCII art of shapes, etc.
        var baseReport = Format(exception);
        var detailed = new StringBuilder(baseReport);

        detailed.AppendLine();
        detailed.AppendLine("Additional Details:");
        detailed.AppendLine();

        // Add shape visualizations
        if (exception.InputShapes != null && exception.InputShapes.Any())
        {
            detailed.AppendLine("Shape Visualization:");
            for (int i = 0; i < exception.InputShapes.Count(); i++)
            {
                var shape = exception.InputShapes.ElementAt(i);
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

    public static string FormatSummary(ShapeMismatchException exception)
    {
        return $"Shape mismatch in '{exception.LayerName}' ({exception.OperationType}): " +
               $"{exception.ProblemDescription}";
    }

    private static string FormatShape(long[] shape)
    {
        if (shape == null) return "null";

        var formatted = shape.Select(d => d == -1 ? "?" : d.ToString());
        return $"[{string.Join(", ", formatted)}]";
    }

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

    private static string VisualizeShape(long[] shape)
    {
        if (shape == null || shape.Length == 0)
            return "  (empty)";

        var sb = new StringBuilder();
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
            sb.Append($" {label} ".PadLeft(shape[i].ToString().Length + 2));
            if (i < shape.Length - 1)
            {
                sb.Append("    ");
            }
        }

        return sb.ToString();
    }

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

    private static string AnalyzeDimensions(ShapeMismatchException exception)
    {
        var analysis = new StringBuilder();

        if (exception.InputShapes == null || !exception.InputShapes.Any())
            return "  No shape information available";

        // Analyze dimension mismatches
        for (int i = 0; i < exception.InputShapes.Count() - 1; i++)
        {
            var shape1 = exception.InputShapes.ElementAt(i);
            var shape2 = exception.InputShapes.ElementAt(i + 1);

            if (shape1 == null || shape2 == null)
                continue;

            var minDim = Math.Min(shape1.Length, shape2.Length);

            for (int d = 0; d < minDim; d++)
            {
                if (shape1[d] != shape2[d])
                {
                    analysis.AppendLine($"  Dimension {d}: Tensor {i} has {shape1[d]}, Tensor {i + 1} has {shape2[d]}");
                    analysis.AppendLine($"    → Mismatch: {Math.Abs(shape1[d] - shape2[d])} difference");

                    if (shape1[d] % shape2[d] == 0 || shape2[d] % shape1[d] == 0)
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
```

### Alternative Format Methods

#### HTML Report Generator
```csharp
public static class HtmlReportFormatter
{
    public static string GenerateReport(ShapeMismatchException exception, string cssPath = null)
    {
        var sb = new StringBuilder();

        sb.AppendLine("<!DOCTYPE html>");
        sb.AppendLine("<html lang='en'>");
        sb.AppendLine("<head>");
        sb.AppendLine("  <meta charset='UTF-8'>");
        sb.AppendLine("  <title>Shape Mismatch Diagnostic Report</title>");

        if (!string.IsNullOrEmpty(cssPath))
        {
            sb.AppendLine($"  <link rel='stylesheet' href='{cssPath}'>");
        }
        else
        {
            sb.AppendLine("<style>");
            sb.AppendLine("  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; }");
            sb.AppendLine("  .header { background: #e74c3c; color: white; padding: 15px; border-radius: 5px; }");
            sb.AppendLine("  .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }");
            sb.AppendLine("  .shape { font-family: monospace; background: #f8f9fa; padding: 5px 10px; border-radius: 3px; }");
            sb.AppendLine("  .error { color: #e74c3c; }");
            sb.AppendLine("  .warning { color: #f39c12; }");
            sb.AppendLine("  .suggestion { color: #27ae60; margin: 5px 0; }");
            sb.AppendLine("  table { border-collapse: collapse; width: 100%; }");
            sb.AppendLine("  td, th { border: 1px solid #ddd; padding: 8px; text-align: left; }");
            sb.AppendLine("  th { background-color: #f2f2f2; }");
            sb.AppendLine("</style>");
        }

        sb.AppendLine("</head>");
        sb.AppendLine("<body>");

        // Header
        sb.AppendLine("  <div class='header'>");
        sb.AppendLine("    <h1>Shape Mismatch Error</h1>");
        sb.AppendLine($"    <p>Layer: {exception.LayerName} | Operation: {exception.OperationType}</p>");
        sb.AppendLine("  </div>");

        // Input Shapes Table
        if (exception.InputShapes != null && exception.InputShapes.Any())
        {
            sb.AppendLine("  <div class='section'>");
            sb.AppendLine("    <h2>Input Shapes</h2>");
            sb.AppendLine("    <table>");
            sb.AppendLine("      <tr><th>Tensor</th><th>Shape</th></tr>");

            for (int i = 0; i < exception.InputShapes.Count(); i++)
            {
                var shape = exception.InputShapes.ElementAt(i);
                var label = ShapeReportFormatter.GetShapeLabel(exception.OperationType, i);
                sb.AppendLine($"      <tr><td>{label}</td><td class='shape'>[{string.Join(", ", shape)}]</td></tr>");
            }

            sb.AppendLine("    </table>");
            sb.AppendLine("  </div>");
        }

        // Problem Description
        sb.AppendLine("  <div class='section'>");
        sb.AppendLine("    <h2>Problem Description</h2>");
        sb.AppendLine($"    <p class='error'>{exception.ProblemDescription}</p>");
        sb.AppendLine("  </div>");

        // Suggested Fixes
        if (exception.SuggestedFixes != null && exception.SuggestedFixes.Any())
        {
            sb.AppendLine("  <div class='section'>");
            sb.AppendLine("    <h2>Suggested Fixes</h2>");

            foreach (var fix in exception.SuggestedFixes)
            {
                sb.AppendLine($"    <div class='suggestion'>{fix}</div>");
            }

            sb.AppendLine("  </div>");
        }

        sb.AppendLine("</body>");
        sb.AppendLine("</html>");

        return sb.ToString();
    }
}
```

#### JSON Export
```csharp
public static class JsonReportFormatter
{
    public static string ToJson(ShapeMismatchException exception)
    {
        var data = new
        {
            layerName = exception.LayerName,
            operationType = exception.OperationType.ToString(),
            batchSize = exception.BatchSize,
            inputShapes = exception.InputShapes?.Select(s => new
            {
                dimensions = s,
                size = s.Aggregate(1L, (a, b) => a * b)
            }),
            expectedShapes = exception.ExpectedShapes?.Select(s => new
            {
                dimensions = s
            }),
            problemDescription = exception.ProblemDescription,
            suggestedFixes = exception.SuggestedFixes,
            timestamp = DateTime.UtcNow.ToString("o"),
            stackTrace = exception.StackTrace
        };

        return System.Text.Json.JsonSerializer.Serialize(
            data,
            new JsonSerializerOptions { WriteIndented = true });
    }
}
```

## Deliverables
- File: `src/Diagnostics/ShapeReportFormatter.cs`
- File: `src/Diagnostics/HtmlReportFormatter.cs`
- File: `src/Diagnostics/JsonReportFormatter.cs`
- Update: `src/Exceptions/ShapeMismatchException.cs` (add new methods)

## Testing Requirements
Create unit tests in `tests/Diagnostics/ShapeReportFormatterTests.cs`:
- Test Format() output for various exceptions
- Test FormatDetailed() includes visualization
- Test FormatSummary() produces one-line summary
- Test FormatShape() with various shapes (including -1)
- Test GetShapeLabel() for different operation types
- Test VisualizeShape() produces correct ASCII art
- Test AnalyzeDimensions() detects mismatches correctly

Create tests in `tests/Diagnostics/HtmlReportFormatterTests.cs`:
- Test HTML generation is valid
- Test HTML includes all necessary sections
- Test HTML with custom CSS path
- Test JSON output structure
- Test JSON serialization includes all fields

## Notes
- Use StringBuilder for efficient string building
- Provide consistent formatting across all report types
- Ensure HTML is properly escaped
- Consider adding color coding for terminals (ANSI codes)
- Make report generation extensible (e.g., custom formatters)
