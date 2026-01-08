using MLFramework.Shapes;
using MLFramework.Diagnostics;

namespace MLFramework.Tests.Diagnostics;

/// <summary>
/// Unit tests for the ErrorReportGenerator class.
/// </summary>
[TestFixture]
public class ErrorReportGeneratorTests
{
    [Test]
    public void GenerateReport_TextFormat_ContainsAllRequiredSections()
    {
        // Arrange
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(
                new SymbolicDimension("batch", 32),
                new SymbolicDimension("input_features", 128))
        };
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(
                new SymbolicDimension("batch", 32),
                new SymbolicDimension("input_features", 256))
        };
        var exception = new ShapeMismatchException(
            "Matrix multiplication",
            expectedShapes,
            actualShapes,
            "Dimension mismatch for fc2 layer");

        // Act
        var report = ErrorReportGenerator.GenerateReport(exception, ErrorReportFormat.Text);

        // Assert
        Assert.That(report, Does.Contain("ML Framework Shape Mismatch Error"));
        Assert.That(report, Does.Contain("Operation: Matrix multiplication"));
        Assert.That(report, Does.Contain("INPUT SHAPES:"));
        Assert.That(report, Does.Contain("EXPECTED SHAPES:"));
        Assert.That(report, Does.Contain("PROBLEM:"));
        Assert.That(report, Does.Contain("CONTEXT:"));
        Assert.That(report, Does.Contain("Dimension mismatch for fc2 layer"));
    }

    [Test]
    public void GenerateReport_TextFormat_FormatsShapesCorrectly()
    {
        // Arrange
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("features", 256)),
            new SymbolicShape(new SymbolicDimension("features", 128), new SymbolicDimension("output", 10))
        };
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("features", 256)),
            new SymbolicShape(new SymbolicDimension("features", 128), new SymbolicDimension("output", 10))
        };
        var exception = new ShapeMismatchException(
            "Linear",
            expectedShapes,
            actualShapes);

        // Act
        var report = ErrorReportGenerator.GenerateReport(exception, ErrorReportFormat.Text);

        // Assert
        Assert.That(report, Does.Contain("[batch, 256]"));
        Assert.That(report, Does.Contain("[features, 10]"));
    }

    [Test]
    public void GenerateReport_MarkdownFormat_ContainsMarkdownHeaders()
    {
        // Arrange
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 32))
        };
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 64))
        };
        var exception = new ShapeMismatchException(
            "BatchNorm",
            expectedShapes,
            actualShapes,
            "Batch size mismatch");

        // Act
        var report = ErrorReportGenerator.GenerateReport(exception, ErrorReportFormat.Markdown);

        // Assert
        Assert.That(report, Does.Contain("# ML Framework Shape Mismatch Error"));
        Assert.That(report, Does.Contain("## Operation"));
        Assert.That(report, Does.Contain("### Input Shapes"));
        Assert.That(report, Does.Contain("### Expected Shapes"));
        Assert.That(report, Does.Contain("## Problem"));
        Assert.That(report, Does.Contain("### Context"));
    }

    [Test]
    public void GenerateReport_MarkdownFormat_UsesBackticksForShapes()
    {
        // Arrange
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("seq", 128))
        };
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("seq", 256))
        };
        var exception = new ShapeMismatchException(
            "SelfAttention",
            expectedShapes,
            actualShapes);

        // Act
        var report = ErrorReportGenerator.GenerateReport(exception, ErrorReportFormat.Markdown);

        // Assert
        Assert.That(report, Does.Contain("`[batch, 128]`"));
        Assert.That(report, Does.Contain("`[batch, 256]`"));
    }

    [Test]
    public void GenerateReport_HtmlFormat_ContainsHtmlStructure()
    {
        // Arrange
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("x", 10))
        };
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("x", 5))
        };
        var exception = new ShapeMismatchException(
            "Reshape",
            expectedShapes,
            actualShapes);

        // Act
        var report = ErrorReportGenerator.GenerateReport(exception, ErrorReportFormat.Html);

        // Assert
        Assert.That(report, Does.StartWith("<!DOCTYPE html>"));
        Assert.That(report, Does.Contain("<html>"));
        Assert.That(report, Does.Contain("<head>"));
        Assert.That(report, Does.Contain("<body>"));
        Assert.That(report, Does.EndWith("</html>"));
    }

    [Test]
    public void GenerateReport_HtmlFormat_ContainsCssStyles()
    {
        // Arrange
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("x", 10))
        };
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("x", 5))
        };
        var exception = new ShapeMismatchException(
            "Reshape",
            expectedShapes,
            actualShapes);

        // Act
        var report = ErrorReportGenerator.GenerateReport(exception, ErrorReportFormat.Html);

        // Assert
        Assert.That(report, Does.Contain(".error-report"));
        Assert.That(report, Does.Contain(".header"));
        Assert.That(report, Does.Contain(".section"));
        Assert.That(report, Does.Contain(".shape"));
        Assert.That(report, Does.Contain(".problem"));
        Assert.That(report, Does.Contain("background: #ffeeee;"));
        Assert.That(report, Does.Contain("color: red;"));
    }

    [Test]
    public void GenerateReport_WithMultipleShapes_DoesNotCrash()
    {
        // Arrange
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64)),
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64))
        };
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64)),
            new SymbolicShape(new SymbolicDimension("batch", 16), new SymbolicDimension("feat", 64))
        };
        var exception = new ShapeMismatchException(
            "Concat",
            expectedShapes,
            actualShapes,
            "Batch size mismatch in second tensor");

        // Act & Assert - Should not throw
        var textReport = ErrorReportGenerator.GenerateReport(exception, ErrorReportFormat.Text);
        var markdownReport = ErrorReportGenerator.GenerateReport(exception, ErrorReportFormat.Markdown);
        var htmlReport = ErrorReportGenerator.GenerateReport(exception, ErrorReportFormat.Html);

        Assert.That(textReport, Is.Not.Null);
        Assert.That(markdownReport, Is.Not.Null);
        Assert.That(htmlReport, Is.Not.Null);
    }

    [Test]
    public void GenerateReport_WithPartialData_HandlesMissingContext()
    {
        // Arrange
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("features", 128))
        };
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("features", 256))
        };
        var exception = new ShapeMismatchException(
            "Linear",
            expectedShapes,
            actualShapes,
            ""); // Empty details

        // Act
        var report = ErrorReportGenerator.GenerateReport(exception, ErrorReportFormat.Text);

        // Assert
        Assert.That(report, Does.Contain("ML Framework Shape Mismatch Error"));
        Assert.That(report, Does.Contain("Operation: Linear"));
        // Should not crash even with empty details
    }

    [Test]
    public void GenerateReport_WithEmptyShapeLists_HandlesGracefully()
    {
        // Arrange
        var expectedShapes = new List<SymbolicShape>();
        var actualShapes = new List<SymbolicShape>();
        var exception = new ShapeMismatchException(
            "UnknownOperation",
            expectedShapes,
            actualShapes,
            "No shape information available");

        // Act
        var report = ErrorReportGenerator.GenerateReport(exception, ErrorReportFormat.Text);

        // Assert
        Assert.That(report, Is.Not.Null);
        Assert.That(report, Does.Contain("UnknownOperation"));
        Assert.That(report, Does.Contain("No shape information available"));
    }

    [Test]
    public void GenerateReport_TextFormat_HasProperLineSeparators()
    {
        // Arrange
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("x", 10))
        };
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("x", 5))
        };
        var exception = new ShapeMismatchException(
            "TestOp",
            expectedShapes,
            actualShapes);

        // Act
        var report = ErrorReportGenerator.GenerateReport(exception, ErrorReportFormat.Text);

        // Assert
        var lines = report.Split('\n');
        Assert.That(lines.Length, Is.GreaterThan(5)); // Should have multiple lines
        Assert.That(report, Does.Contain("================================================================"));
    }

    [Test]
    public void GenerateReport_WithNullException_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
        {
            ErrorReportGenerator.GenerateReport(null!, ErrorReportFormat.Text);
        });
    }

    [Test]
    public void GenerateReport_WithInvalidFormat_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "Test",
            new List<SymbolicShape>(),
            new List<SymbolicShape>());

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
        {
            ErrorReportGenerator.GenerateReport(exception, (ErrorReportFormat)999);
        });
    }

    [Test]
    public void GenerateReport_OutputIsReadable_Comprehensive()
    {
        // Arrange
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("input", 256)),
            new SymbolicShape(new SymbolicDimension("input", 128), new SymbolicDimension("output", 10))
        };
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("input", 256)),
            new SymbolicShape(new SymbolicDimension("input", 256), new SymbolicDimension("output", 10))
        };
        var exception = new ShapeMismatchException(
            "Matrix multiplication",
            expectedShapes,
            actualShapes,
            "Dimension 1 mismatch: expected 128, got 256");

        // Act
        var report = ErrorReportGenerator.GenerateReport(exception, ErrorReportFormat.Text);

        // Assert - Verify report contains actionable information
        Assert.That(report, Does.Contain("Matrix multiplication"));
        Assert.That(report, Does.Contain("[batch, 256]"));
        Assert.That(report, Does.Contain("[input, 128]"));
        Assert.That(report, Does.Contain("[input, 256]"));
        Assert.That(report, Does.Contain("[output, 10]"));
        Assert.That(report, Does.Contain("Dimension 1 mismatch"));
    }

    [Test]
    public void GenerateReport_DefaultFormat_UsesTextFormat()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "Test",
            new List<SymbolicShape>(),
            new List<SymbolicShape>());

        // Act
        var report = ErrorReportGenerator.GenerateReport(exception);

        // Assert - Should use text format by default
        Assert.That(report, Does.Contain("================================================================"));
    }
}
