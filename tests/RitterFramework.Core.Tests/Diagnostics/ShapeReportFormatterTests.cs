using RitterFramework.Core;
using RitterFramework.Core.Diagnostics;
using Xunit;

namespace RitterFramework.Core.Tests.Diagnostics;

public class ShapeReportFormatterTests
{
    [Fact]
    public void Format_GeneratesBasicReport()
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
        var report = Diagnostics.ShapeReportFormatter.Format(exception);

        // Assert
        Assert.Contains("SHAPE MISMATCH DIAGNOSTIC REPORT", report);
        Assert.Contains("Layer: Linear", report);
        Assert.Contains("Operation: MatrixMultiply", report);
        Assert.Contains("Input: [32, 128]", report);
        Assert.Contains("Weight: [256, 64]", report);
        Assert.Contains("Expected 0: [32, 64]", report);
        Assert.Contains("Problem:", report);
        Assert.Contains("Suggested Fixes:", report);
    }

    [Fact]
    public void FormatDetailed_IncludesVisualizationAndAnalysis()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Conv2",
            operationType: OperationType.Conv2D,
            inputShapes: new List<long[]> { new long[] { 16, 3, 32, 32 }, new long[] { 64, 3, 3, 3 } },
            expectedShapes: new List<long[]> { new long[] { 16, 64, 30, 30 } },
            problemDescription: "Kernel dimensions mismatch"
        );

        // Act
        var report = Diagnostics.ShapeReportFormatter.FormatDetailed(exception);

        // Assert
        Assert.Contains("SHAPE MISMATCH DIAGNOSTIC REPORT", report);
        Assert.Contains("Additional Details:", report);
        Assert.Contains("Shape Visualization:", report);
        Assert.Contains("Dimension Analysis:", report);
        Assert.Contains("→", report); // ASCII visualization arrow
    }

    [Fact]
    public void FormatSummary_GeneratesOneLineSummary()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Linear",
            operationType: OperationType.MatrixMultiply,
            inputShapes: new List<long[]>(),
            expectedShapes: new List<long[]>(),
            problemDescription: "Dimension mismatch in matrix multiplication"
        );

        // Act
        var summary = Diagnostics.ShapeReportFormatter.FormatSummary(exception);

        // Assert
        Assert.Contains("Shape mismatch in 'Linear' (MatrixMultiply)", summary);
        Assert.Contains("Dimension mismatch in matrix multiplication", summary);
    }

    [Fact]
    public void FormatShape_HandlesNegativeOne()
    {
        // Arrange
        var shape = new long[] { 1, -1, 28, 28 };

        // Act - using reflection to test private method
        // Instead, we'll test through Format which uses FormatShape internally
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Flatten",
            operationType: OperationType.Flatten,
            inputShapes: new List<long[]> { shape },
            expectedShapes: new List<long[]> { new long[] { 1, 784 } },
            problemDescription: "Flattening issue"
        );

        var report = Diagnostics.ShapeReportFormatter.Format(exception);

        // Assert
        Assert.Contains("[1, ?, 28, 28]", report);
    }

    [Fact]
    public void FormatShape_HandlesNullShape()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Unknown",
            operationType: OperationType.Unknown,
            inputShapes: new List<long[]>(),
            expectedShapes: new List<long[]>(),
            problemDescription: "Test null handling"
        );

        // Act
        var report = Diagnostics.ShapeReportFormatter.Format(exception);

        // Assert
        Assert.DoesNotContain("Input Shapes:", report);
    }

    [Theory]
    [InlineData(OperationType.MatrixMultiply, 0, "Input")]
    [InlineData(OperationType.MatrixMultiply, 1, "Weight")]
    [InlineData(OperationType.Conv2D, 0, "Input")]
    [InlineData(OperationType.Conv2D, 1, "Kernel")]
    [InlineData(OperationType.Conv1D, 0, "Input")]
    [InlineData(OperationType.Conv1D, 1, "Kernel")]
    [InlineData(OperationType.Concat, 0, "Input 0")]
    [InlineData(OperationType.Concat, 1, "Input 1")]
    [InlineData(OperationType.Stack, 0, "Input 0")]
    [InlineData(OperationType.Unknown, 0, "Tensor 0")]
    public void GetShapeLabel_ReturnsCorrectLabel(OperationType op, int index, string expectedLabel)
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Test",
            operationType: op,
            inputShapes: new List<long[]> { new long[] { 1, 1 }, new long[] { 1, 1 } },
            expectedShapes: new List<long[]>(),
            problemDescription: "Test"
        );

        // Act
        var report = Diagnostics.ShapeReportFormatter.Format(exception);

        // Assert
        if (index == 0)
        {
            Assert.Contains($"{expectedLabel}: [1, 1]", report);
        }
        else
        {
            Assert.Contains($"{expectedLabel}: [1, 1]", report);
        }
    }

    [Fact]
    public void VisualizeShape_GeneratesCorrectAsciiArt()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Test",
            operationType: OperationType.Conv2D,
            inputShapes: new List<long[]> { new long[] { 16, 3, 32, 32 } },
            expectedShapes: new List<long[]>(),
            problemDescription: "Test"
        );

        // Act
        var report = Diagnostics.ShapeReportFormatter.FormatDetailed(exception);

        // Assert
        Assert.Contains("[16]", report);
        Assert.Contains("[3]", report);
        Assert.Contains("[32]", report);
        Assert.Contains("→", report);
        Assert.Contains("batch", report);
        Assert.Contains("ch", report);
        Assert.Contains("h", report);
        Assert.Contains("w", report);
    }

    [Fact]
    public void VisualizeShape_Handles2DShape()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Test",
            operationType: OperationType.MatrixMultiply,
            inputShapes: new List<long[]> { new long[] { 128, 256 } },
            expectedShapes: new List<long[]>(),
            problemDescription: "Test"
        );

        // Act
        var report = Diagnostics.ShapeReportFormatter.FormatDetailed(exception);

        // Assert
        Assert.Contains("rows", report);
        Assert.Contains("cols", report);
    }

    [Fact]
    public void AnalyzeDimensions_DetectsMismatches()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Test",
            operationType: OperationType.Concat,
            inputShapes: new List<long[]> { new long[] { 16, 3, 32, 32 }, new long[] { 16, 64, 32, 32 } },
            expectedShapes: new List<long[]>(),
            problemDescription: "Test"
        );

        // Act
        var report = Diagnostics.ShapeReportFormatter.FormatDetailed(exception);

        // Assert
        Assert.Contains("Dimension Analysis:", report);
        Assert.Contains("Dimension 1:", report);
        Assert.Contains("Mismatch:", report);
    }

    [Fact]
    public void AnalyzeDimensions_DetectsMultiples()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Test",
            operationType: OperationType.Unknown,
            inputShapes: new List<long[]> { new long[] { 16, 64 }, new long[] { 16, 32 } },
            expectedShapes: new List<long[]>(),
            problemDescription: "Test"
        );

        // Act
        var report = Diagnostics.ShapeReportFormatter.FormatDetailed(exception);

        // Assert
        Assert.Contains("multiples", report);
        Assert.Contains("broadcast issue", report);
    }

    [Fact]
    public void AnalyzeDimensions_DetectsDifferentDimensionCounts()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Test",
            operationType: OperationType.Unknown,
            inputShapes: new List<long[]> { new long[] { 16, 3, 32, 32 }, new long[] { 16, 64 } },
            expectedShapes: new List<long[]>(),
            problemDescription: "Test"
        );

        // Act
        var report = Diagnostics.ShapeReportFormatter.FormatDetailed(exception);

        // Assert
        Assert.Contains("Dimension count mismatch:", report);
    }

    [Fact]
    public void AnalyzeDimensions_NoObviousMismatches()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            message: "Shape mismatch",
            layerName: "Test",
            operationType: OperationType.Unknown,
            inputShapes: new List<long[]> { new long[] { 16, 3, 32, 32 }, new long[] { 16, 3, 32, 32 } },
            expectedShapes: new List<long[]>(),
            problemDescription: "Test"
        );

        // Act
        var report = Diagnostics.ShapeReportFormatter.FormatDetailed(exception);

        // Assert
        Assert.Contains("No obvious dimension mismatches found", report);
    }

    [Fact]
    public void Format_WithMinimalInformation()
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
        var report = Diagnostics.ShapeReportFormatter.Format(exception);

        // Assert
        Assert.Contains("SHAPE MISMATCH DIAGNOSTIC REPORT", report);
        Assert.Contains("Layer: ", report);
        Assert.Contains("Operation: Unknown", report);
        Assert.Contains("Problem: No problem description provided.", report);
    }
}
