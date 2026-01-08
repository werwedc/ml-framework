using System.Text;

namespace MLFramework.Diagnostics;

/// <summary>
/// Enum specifying the format for error reports.
/// </summary>
public enum ErrorReportFormat
{
    /// <summary>
    /// Plain text format
    /// </summary>
    Text,

    /// <summary>
    /// Markdown format
    /// </summary>
    Markdown,

    /// <summary>
    /// HTML format with basic styling
    /// </summary>
    Html
}

/// <summary>
/// Generates formatted error reports from ShapeMismatchException instances.
/// Supports Text, Markdown, and HTML formats.
/// </summary>
public class ErrorReportGenerator
{
    private const int LineLength = 64;

    /// <summary>
    /// Generates a formatted error report from a ShapeMismatchException.
    /// </summary>
    /// <param name="exception">The exception to generate a report for.</param>
    /// <param name="format">The format of the report (default: Text).</param>
    /// <returns>A formatted error report as a string.</returns>
    public static string GenerateReport(
        Shapes.ShapeMismatchException exception,
        ErrorReportFormat format = ErrorReportFormat.Text)
    {
        if (exception == null)
        {
            throw new ArgumentNullException(nameof(exception));
        }

        return format switch
        {
            ErrorReportFormat.Text => GenerateTextReport(exception),
            ErrorReportFormat.Markdown => GenerateMarkdownReport(exception),
            ErrorReportFormat.Html => GenerateHtmlReport(exception),
            _ => throw new ArgumentOutOfRangeException(nameof(format), format, "Unsupported report format")
        };
    }

    /// <summary>
    /// Generates a plain text format error report.
    /// </summary>
    private static string GenerateTextReport(Shapes.ShapeMismatchException exception)
    {
        var report = new StringBuilder();

        // Header
        report.AppendLine(new string('=', LineLength));
        report.AppendLine("ML Framework Shape Mismatch Error");
        report.AppendLine(new string('=', LineLength));
        report.AppendLine();

        // Operation
        report.AppendLine($"Operation: {exception.OperationName}");
        report.AppendLine();

        // Input Shapes
        if (exception.ActualShapes.Count > 0)
        {
            report.AppendLine("INPUT SHAPES:");
            for (int i = 0; i < exception.ActualShapes.Count; i++)
            {
                report.AppendLine($"  Tensor {i + 1}: {FormatShape(exception.ActualShapes[i])}");
            }
            report.AppendLine();
        }

        // Expected Shapes
        if (exception.ExpectedShapes.Count > 0)
        {
            report.AppendLine("EXPECTED SHAPES:");
            for (int i = 0; i < exception.ExpectedShapes.Count; i++)
            {
                report.AppendLine($"  Tensor {i + 1}: {FormatShape(exception.ExpectedShapes[i])}");
            }
            report.AppendLine();
        }

        // Problem
        report.AppendLine("PROBLEM:");
        report.AppendLine($"  {exception.Message}");
        report.AppendLine();

        // Details
        if (!string.IsNullOrWhiteSpace(exception.Details))
        {
            report.AppendLine("CONTEXT:");
            report.AppendLine($"  {exception.Details}");
            report.AppendLine();
        }

        // Footer
        report.AppendLine(new string('=', LineLength));

        return report.ToString();
    }

    /// <summary>
    /// Generates a Markdown format error report.
    /// </summary>
    private static string GenerateMarkdownReport(Shapes.ShapeMismatchException exception)
    {
        var report = new StringBuilder();

        // Header
        report.AppendLine("# ML Framework Shape Mismatch Error");
        report.AppendLine();

        // Operation
        report.AppendLine("## Operation");
        report.AppendLine(exception.OperationName);
        report.AppendLine();

        // Input Shapes
        if (exception.ActualShapes.Count > 0)
        {
            report.AppendLine("### Input Shapes");
            for (int i = 0; i < exception.ActualShapes.Count; i++)
            {
                report.AppendLine($"- Tensor {i + 1}: `{FormatShape(exception.ActualShapes[i])}`");
            }
            report.AppendLine();
        }

        // Expected Shapes
        if (exception.ExpectedShapes.Count > 0)
        {
            report.AppendLine("### Expected Shapes");
            for (int i = 0; i < exception.ExpectedShapes.Count; i++)
            {
                report.AppendLine($"- Tensor {i + 1}: `{FormatShape(exception.ExpectedShapes[i])}`");
            }
            report.AppendLine();
        }

        // Problem
        report.AppendLine("## Problem");
        report.AppendLine(exception.Message);
        report.AppendLine();

        // Details
        if (!string.IsNullOrWhiteSpace(exception.Details))
        {
            report.AppendLine("### Context");
            report.AppendLine(exception.Details);
            report.AppendLine();
        }

        return report.ToString();
    }

    /// <summary>
    /// Generates an HTML format error report with basic styling.
    /// </summary>
    private static string GenerateHtmlReport(Shapes.ShapeMismatchException exception)
    {
        var html = new StringBuilder();

        html.AppendLine("<!DOCTYPE html>");
        html.AppendLine("<html>");
        html.AppendLine("<head>");
        html.AppendLine("    <style>");
        html.AppendLine("        .error-report { font-family: monospace; padding: 20px; background: #ffeeee; margin: 20px; }");
        html.AppendLine("        .header { font-size: 18px; font-weight: bold; margin-bottom: 10px; }");
        html.AppendLine("        .section { margin: 10px 0; }");
        html.AppendLine("        .shape { background: #ffffee; padding: 5px; margin: 2px 0; }");
        html.AppendLine("        .problem { color: red; font-weight: bold; }");
        html.AppendLine("        .label { font-weight: bold; }");
        html.AppendLine("    </style>");
        html.AppendLine("</head>");
        html.AppendLine("<body>");
        html.AppendLine("    <div class=\"error-report\">");

        // Header
        html.AppendLine("        <div class=\"header\">ML Framework Shape Mismatch Error</div>");

        // Operation
        html.AppendLine("        <div class=\"section\">");
        html.AppendLine($"            <span class=\"label\">Operation:</span> {exception.OperationName}<br>");
        html.AppendLine("        </div>");

        // Input Shapes
        if (exception.ActualShapes.Count > 0)
        {
            html.AppendLine("        <div class=\"section\">");
            html.AppendLine("            <span class=\"label\">Input Shapes:</span><br>");
            for (int i = 0; i < exception.ActualShapes.Count; i++)
            {
                html.AppendLine($"            <div class=\"shape\">Tensor {i + 1}: {FormatShape(exception.ActualShapes[i])}</div>");
            }
            html.AppendLine("        </div>");
        }

        // Expected Shapes
        if (exception.ExpectedShapes.Count > 0)
        {
            html.AppendLine("        <div class=\"section\">");
            html.AppendLine("            <span class=\"label\">Expected Shapes:</span><br>");
            for (int i = 0; i < exception.ExpectedShapes.Count; i++)
            {
                html.AppendLine($"            <div class=\"shape\">Tensor {i + 1}: {FormatShape(exception.ExpectedShapes[i])}</div>");
            }
            html.AppendLine("        </div>");
        }

        // Problem
        html.AppendLine("        <div class=\"section\">");
        html.AppendLine("            <span class=\"label\">Problem:</span><br>");
        html.AppendLine($"            <span class=\"problem\">{exception.Message}</span>");
        html.AppendLine("        </div>");

        // Details
        if (!string.IsNullOrWhiteSpace(exception.Details))
        {
            html.AppendLine("        <div class=\"section\">");
            html.AppendLine("            <span class=\"label\">Context:</span><br>");
            html.AppendLine($"            {exception.Details}");
            html.AppendLine("        </div>");
        }

        html.AppendLine("    </div>");
        html.AppendLine("</body>");
        html.AppendLine("</html>");

        return html.ToString();
    }

    /// <summary>
    /// Formats a SymbolicShape as a string.
    /// </summary>
    private static string FormatShape(Shapes.SymbolicShape shape)
    {
        if (shape == null)
        {
            return "[null]";
        }

        return shape.ToString();
    }
}
