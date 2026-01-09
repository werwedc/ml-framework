using System.Web;

namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Generates HTML formatted diagnostic reports from shape mismatch exceptions.
/// </summary>
public static class HtmlReportFormatter
{
    /// <summary>
    /// Generates an HTML report from a shape mismatch exception.
    /// </summary>
    /// <param name="exception">The exception to format.</param>
    /// <param name="cssPath">Optional path to an external CSS file.</param>
    /// <returns>An HTML formatted report.</returns>
    public static string GenerateReport(ShapeMismatchException exception, string? cssPath = null)
    {
        var sb = new System.Text.StringBuilder();

        sb.AppendLine("<!DOCTYPE html>");
        sb.AppendLine("<html lang='en'>");
        sb.AppendLine("<head>");
        sb.AppendLine("  <meta charset='UTF-8'>");
        sb.AppendLine("  <title>Shape Mismatch Diagnostic Report</title>");

        if (!string.IsNullOrEmpty(cssPath))
        {
            sb.AppendLine($"  <link rel='stylesheet' href='{HtmlEncode(cssPath)}'>");
        }
        else
        {
            sb.AppendLine("<style>");
            sb.AppendLine("  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }");
            sb.AppendLine("  .container { max-width: 900px; margin: 0 auto; }");
            sb.AppendLine("  .header { background: #e74c3c; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }");
            sb.AppendLine("  .section { background: white; margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }");
            sb.AppendLine("  .section h2 { margin-top: 0; color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }");
            sb.AppendLine("  .shape { font-family: 'Consolas', 'Monaco', monospace; background: #f8f9fa; padding: 8px 12px; border-radius: 4px; border: 1px solid #e9ecef; }");
            sb.AppendLine("  .error { color: #e74c3c; font-weight: bold; }");
            sb.AppendLine("  .warning { color: #f39c12; }");
            sb.AppendLine("  .suggestion { color: #27ae60; margin: 8px 0; padding: 8px; background: #d5f4e6; border-left: 4px solid #27ae60; }");
            sb.AppendLine("  table { border-collapse: collapse; width: 100%; margin: 10px 0; }");
            sb.AppendLine("  td, th { border: 1px solid #ddd; padding: 12px; text-align: left; }");
            sb.AppendLine("  th { background-color: #3498db; color: white; }");
            sb.AppendLine("  tr:nth-child(even) { background-color: #f2f2f2; }");
            sb.AppendLine("  .label { font-weight: bold; color: #555; }");
            sb.AppendLine("</style>");
        }

        sb.AppendLine("</head>");
        sb.AppendLine("<body>");
        sb.AppendLine("<div class='container'>");

        // Header
        sb.AppendLine("  <div class='header'>");
        sb.AppendLine("    <h1>Shape Mismatch Error</h1>");
        sb.AppendLine($"    <p>Layer: <strong>{HtmlEncode(exception.LayerName ?? "Unknown")}</strong> | Operation: <strong>{exception.OperationType}</strong></p>");
        sb.AppendLine("  </div>");

        // Input Shapes Table
        if (exception.InputShapes != null && exception.InputShapes.Count > 0)
        {
            sb.AppendLine("  <div class='section'>");
            sb.AppendLine("    <h2>Input Shapes</h2>");
            sb.AppendLine("    <table>");
            sb.AppendLine("      <tr><th>Tensor</th><th>Shape</th></tr>");

            for (int i = 0; i < exception.InputShapes.Count; i++)
            {
                var shape = exception.InputShapes[i];
                var label = GetShapeLabel(exception.OperationType, i);
                var shapeStr = shape != null ? $"[{string.Join(", ", shape)}]" : "null";
                sb.AppendLine($"      <tr><td><span class='label'>{HtmlEncode(label)}</span></td><td class='shape'>{HtmlEncode(shapeStr)}</td></tr>");
            }

            sb.AppendLine("    </table>");
            sb.AppendLine("  </div>");
        }

        // Expected Shapes Table
        if (exception.ExpectedShapes != null && exception.ExpectedShapes.Count > 0)
        {
            sb.AppendLine("  <div class='section'>");
            sb.AppendLine("    <h2>Expected Shapes</h2>");
            sb.AppendLine("    <table>");
            sb.AppendLine("      <tr><th>Index</th><th>Shape</th></tr>");

            for (int i = 0; i < exception.ExpectedShapes.Count; i++)
            {
                var shape = exception.ExpectedShapes[i];
                var shapeStr = shape != null ? $"[{string.Join(", ", shape)}]" : "null";
                sb.AppendLine($"      <tr><td><span class='label'>Expected {i}</span></td><td class='shape'>{HtmlEncode(shapeStr)}</td></tr>");
            }

            sb.AppendLine("    </table>");
            sb.AppendLine("  </div>");
        }

        // Problem Description
        sb.AppendLine("  <div class='section'>");
        sb.AppendLine("    <h2>Problem Description</h2>");
        sb.AppendLine($"    <p class='error'>{HtmlEncode(exception.ProblemDescription ?? "No description provided")}</p>");
        sb.AppendLine("  </div>");

        // Suggested Fixes
        if (exception.SuggestedFixes != null && exception.SuggestedFixes.Count > 0)
        {
            sb.AppendLine("  <div class='section'>");
            sb.AppendLine("    <h2>Suggested Fixes</h2>");

            foreach (var fix in exception.SuggestedFixes)
            {
                sb.AppendLine($"    <div class='suggestion'>{HtmlEncode(fix)}</div>");
            }

            sb.AppendLine("  </div>");
        }

        // Footer
        sb.AppendLine("  <div class='section'>");
        sb.AppendLine("    <p style='color: #7f8c8d; text-align: center;'>");
        sb.AppendLine($"      Generated at: {System.DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
        sb.AppendLine("    </p>");
        sb.AppendLine("  </div>");

        sb.AppendLine("</div>");
        sb.AppendLine("</body>");
        sb.AppendLine("</html>");

        return sb.ToString();
    }

    /// <summary>
    /// Gets a descriptive label for a shape based on the operation type and index.
    /// </summary>
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
    /// HTML-encodes a string to prevent XSS.
    /// </summary>
    private static string HtmlEncode(string? input)
    {
        if (string.IsNullOrEmpty(input))
            return string.Empty;

        return System.Web.HttpUtility.HtmlEncode(input);
    }
}
