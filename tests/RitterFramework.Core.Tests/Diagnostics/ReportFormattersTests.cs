using RitterFramework.Core;
using RitterFramework.Core.Diagnostics;
using System.Text.Json;
using Xunit;

namespace RitterFramework.Core.Tests.Diagnostics;

public class ReportFormattersTests
{
    [Fact]
    public void HtmlReportFormatter_GeneratesValidHtml()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Linear",
            operationType: OperationType.MatrixMultiply,
            inputShapes: new List<long[]> { new long[] { 32, 128 }, new long[] { 256, 64 } },
            expectedShapes: new List<long[]> { new long[] { 32, 64 } },
            problemDescription: "Dimension 1 of input (128) does not match dimension 0 of weight (256)",
            suggestedFixes: new List<string> { "Adjust input dimensions", "Adjust weight matrix size" }
        );

        // Act
        var html = HtmlReportFormatter.GenerateReport(exception);

        // Assert
        Assert.Contains("<!DOCTYPE html>", html);
        Assert.Contains("<html lang='en'>", html);
        Assert.Contains("<head>", html);
        Assert.Contains("<body>", html);
        Assert.Contains("Shape Mismatch Error", html);
        Assert.Contains("Layer:", html);
        Assert.Contains("Linear", html);
        Assert.Contains("MatrixMultiply", html);
        Assert.Contains("Input Shapes", html);
        Assert.Contains("Expected Shapes", html);
        Assert.Contains("Problem Description", html);
        Assert.Contains("Suggested Fixes", html);
    }

    [Fact]
    public void HtmlReportFormatter_IncludesStyles()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Test",
            operationType: OperationType.Unknown,
            inputShapes: new List<long[]>(),
            expectedShapes: new List<long[]>(),
            problemDescription: "Test"
        );

        // Act
        var html = HtmlReportFormatter.GenerateReport(exception);

        // Assert
        Assert.Contains("<style>", html);
        Assert.Contains("</style>", html);
        Assert.Contains("font-family:", html);
        Assert.Contains(".header", html);
        Assert.Contains(".section", html);
        Assert.Contains(".shape", html);
    }

    [Fact]
    public void HtmlReportFormatter_IncludesInputShapesTable()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Conv2",
            operationType: OperationType.Conv2D,
            inputShapes: new List<long[]> { new long[] { 16, 3, 32, 32 }, new long[] { 64, 3, 3, 3 } },
            expectedShapes: new List<long[]>(),
            problemDescription: "Test"
        );

        // Act
        var html = HtmlReportFormatter.GenerateReport(exception);

        // Assert
        Assert.Contains("<table>", html);
        Assert.Contains("<th>Tensor</th>", html);
        Assert.Contains("<th>Shape</th>", html);
        Assert.Contains("Input", html);
        Assert.Contains("Kernel", html);
        Assert.Contains("[16, 3, 32, 32]", html);
        Assert.Contains("[64, 3, 3, 3]", html);
    }

    [Fact]
    public void HtmlReportFormatter_IncludesExpectedShapesTable()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Test",
            operationType: OperationType.Unknown,
            inputShapes: new List<long[]>(),
            expectedShapes: new List<long[]> { new long[] { 32, 64 }, new long[] { 32, 64 } },
            problemDescription: "Test"
        );

        // Act
        var html = HtmlReportFormatter.GenerateReport(exception);

        // Assert
        Assert.Contains("Expected Shapes", html);
        Assert.Contains("Expected 0", html);
        Assert.Contains("[32, 64]", html);
    }

    [Fact]
    public void HtmlReportFormatter_IncludesProblemDescription()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Test",
            operationType: OperationType.Unknown,
            inputShapes: new List<long[]>(),
            expectedShapes: new List<long[]>(),
            problemDescription: "Critical dimension mismatch detected"
        );

        // Act
        var html = HtmlReportFormatter.GenerateReport(exception);

        // Assert
        Assert.Contains("Problem Description", html);
        Assert.Contains("Critical dimension mismatch detected", html);
        Assert.Contains("class='error'", html);
    }

    [Fact]
    public void HtmlReportFormatter_IncludesSuggestedFixes()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Test",
            operationType: OperationType.Unknown,
            inputShapes: new List<long[]>(),
            expectedShapes: new List<long[]>(),
            problemDescription: "Test",
            suggestedFixes: new List<string> { "Fix 1: Adjust dimensions", "Fix 2: Reshape tensor" }
        );

        // Act
        var html = HtmlReportFormatter.GenerateReport(exception);

        // Assert
        Assert.Contains("Suggested Fixes", html);
        Assert.Contains("Fix 1: Adjust dimensions", html);
        Assert.Contains("Fix 2: Reshape tensor", html);
        Assert.Contains("class='suggestion'", html);
    }

    [Fact]
    public void HtmlReportFormatter_AcceptsCustomCssPath()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Test",
            operationType: OperationType.Unknown,
            inputShapes: new List<long[]>(),
            expectedShapes: new List<long[]>(),
            problemDescription: "Test"
        );

        // Act
        var html = HtmlReportFormatter.GenerateReport(exception, "/path/to/custom.css");

        // Assert
        Assert.Contains("<link rel='stylesheet' href='/path/to/custom.css'>", html);
        Assert.DoesNotContain("<style>", html);
    }

    [Fact]
    public void JsonReportFormatter_GeneratesValidJson()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Linear",
            operationType: OperationType.MatrixMultiply,
            inputShapes: new List<long[]> { new long[] { 32, 128 } },
            expectedShapes: new List<long[]> { new long[] { 32, 64 } },
            problemDescription: "Test",
            suggestedFixes: new List<string> { "Fix 1" }
        );

        // Act
        var json = JsonReportFormatter.ToJson(exception);

        // Assert
        Assert.NotNull(json);

        // Verify it's valid JSON
        var jsonDoc = JsonDocument.Parse(json);
        Assert.True(jsonDoc.RootElement.ValueKind == JsonValueKind.Object);
    }

    [Fact]
    public void JsonReportFormatter_IncludesAllFields()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Linear",
            operationType: OperationType.MatrixMultiply,
            inputShapes: new List<long[]> { new long[] { 32, 128 } },
            expectedShapes: new List<long[]> { new long[] { 32, 64 } },
            problemDescription: "Test problem",
            suggestedFixes: new List<string> { "Fix 1", "Fix 2" }
        );

        // Act
        var json = JsonReportFormatter.ToJson(exception);
        var jsonDoc = JsonDocument.Parse(json);
        var root = jsonDoc.RootElement;

        // Assert
        Assert.True(root.TryGetProperty("layerName", out var layerName));
        Assert.Equal("Linear", layerName.GetString());

        Assert.True(root.TryGetProperty("operationType", out var operationType));
        Assert.Equal("MatrixMultiply", operationType.GetString());

        Assert.True(root.TryGetProperty("inputShapes", out var inputShapes));
        Assert.Equal(JsonValueKind.Array, inputShapes.ValueKind);
        Assert.True(inputShapes.GetArrayLength() > 0);

        Assert.True(root.TryGetProperty("expectedShapes", out var expectedShapes));
        Assert.Equal(JsonValueKind.Array, expectedShapes.ValueKind);

        Assert.True(root.TryGetProperty("problemDescription", out var problemDesc));
        Assert.Equal("Test problem", problemDesc.GetString());

        Assert.True(root.TryGetProperty("suggestedFixes", out var fixes));
        Assert.Equal(JsonValueKind.Array, fixes.ValueKind);
        Assert.Equal(2, fixes.GetArrayLength());

        Assert.True(root.TryGetProperty("timestamp", out var timestamp));
        Assert.NotNull(timestamp.GetString());

        Assert.True(root.TryGetProperty("stackTrace", out var stackTrace));
        Assert.NotNull(stackTrace.GetString());
    }

    [Fact]
    public void JsonReportFormatter_CalculatesSize()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Test",
            operationType: OperationType.Unknown,
            inputShapes: new List<long[]> { new long[] { 2, 3, 4, 5 } },
            expectedShapes: new List<long[]>(),
            problemDescription: "Test"
        );

        // Act
        var json = JsonReportFormatter.ToJson(exception);
        var jsonDoc = JsonDocument.Parse(json);
        var root = jsonDoc.RootElement;

        // Assert
        Assert.True(root.TryGetProperty("inputShapes", out var inputShapes));
        var firstShape = inputShapes[0];
        Assert.True(firstShape.TryGetProperty("dimensions", out var dimensions));
        Assert.True(firstShape.TryGetProperty("size", out var size));

        Assert.Equal(120, size.GetInt64()); // 2 * 3 * 4 * 5 = 120
    }

    [Fact]
    public void JsonReportFormatter_HandlesEmptyLists()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: null,
            operationType: OperationType.Unknown,
            inputShapes: new List<long[]>(),
            expectedShapes: new List<long[]>(),
            problemDescription: null,
            suggestedFixes: new List<string>()
        );

        // Act
        var json = JsonReportFormatter.ToJson(exception);
        var jsonDoc = JsonDocument.Parse(json);
        var root = jsonDoc.RootElement;

        // Assert
        Assert.True(root.TryGetProperty("layerName", out var layerName));
        Assert.Null(layerName.GetString());

        Assert.True(root.TryGetProperty("inputShapes", out var inputShapes));
        Assert.Null(inputShapes.ValueKind); // Will be null

        Assert.True(root.TryGetProperty("expectedShapes", out var expectedShapes));
        Assert.Null(expectedShapes.ValueKind); // Will be null

        Assert.True(root.TryGetProperty("problemDescription", out var problemDesc));
        Assert.Null(problemDesc.GetString());

        Assert.True(root.TryGetProperty("suggestedFixes", out var fixes));
        Assert.Null(fixes.ValueKind); // Will be null
    }

    [Fact]
    public void ShapeMismatchException_GetDiagnosticReport_DelegatesToShapeReportFormatter()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Test",
            operationType: OperationType.MatrixMultiply,
            inputShapes: new List<long[]> { new long[] { 32, 128 } },
            expectedShapes: new List<long[]> { new long[] { 32, 64 } },
            problemDescription: "Test"
        );

        // Act
        var report = exception.GetDiagnosticReport();

        // Assert
        Assert.Contains("SHAPE MISMATCH DIAGNOSTIC REPORT", report);
        Assert.Contains("Layer: Test", report);
        Assert.Contains("Operation: MatrixMultiply", report);
    }

    [Fact]
    public void ShapeMismatchException_GetDetailedReport_UsesShapeReportFormatter()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Test",
            operationType: OperationType.MatrixMultiply,
            inputShapes: new List<long[]> { new long[] { 32, 128 } },
            expectedShapes: new List<long[]>(),
            problemDescription: "Test"
        );

        // Act
        var report = exception.GetDetailedReport();

        // Assert
        Assert.Contains("SHAPE MISMATCH DIAGNOSTIC REPORT", report);
        Assert.Contains("Additional Details:", report);
        Assert.Contains("Shape Visualization:", report);
        Assert.Contains("Dimension Analysis:", report);
    }

    [Fact]
    public void ShapeMismatchException_GetSummary_UsesShapeReportFormatter()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Linear",
            operationType: OperationType.MatrixMultiply,
            inputShapes: new List<long[]>(),
            expectedShapes: new List<long[]>(),
            problemDescription: "Dimension mismatch detected"
        );

        // Act
        var summary = exception.GetSummary();

        // Assert
        Assert.Contains("Shape mismatch in 'Linear' (MatrixMultiply)", summary);
        Assert.Contains("Dimension mismatch detected", summary);
    }
}
