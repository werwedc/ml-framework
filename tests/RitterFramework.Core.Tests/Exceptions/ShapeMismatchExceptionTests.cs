using MLFramework.Core;
using MLFramework.Exceptions;
using Xunit;

namespace MLFramework.Tests.Exceptions;

/// <summary>
/// Unit tests for ShapeMismatchException class.
/// </summary>
public class ShapeMismatchExceptionTests
{
    [Fact]
    public void Constructor_WithAllParameters_CreatesExceptionCorrectly()
    {
        // Arrange
        var layerName = "test_layer";
        var operationType = OperationType.MatrixMultiply;
        var inputShapes = new[] { new long[] { 32, 256 } };
        var expectedShapes = new[] { new long[] { 32, 128 } };
        var problemDescription = "Dimension mismatch";
        var suggestedFixes = new List<string> { "Fix 1", "Fix 2" };

        // Act
        var exception = new ShapeMismatchException(
            layerName,
            operationType,
            inputShapes,
            expectedShapes,
            problemDescription,
            suggestedFixes);

        // Assert
        Assert.Equal(layerName, exception.LayerName);
        Assert.Equal(operationType, exception.OperationType);
        Assert.Equal(inputShapes, exception.InputShapes);
        Assert.Equal(expectedShapes, exception.ExpectedShapes);
        Assert.Equal(problemDescription, exception.ProblemDescription);
        Assert.Equal(suggestedFixes, exception.SuggestedFixes);
    }

    [Fact]
    public void Constructor_WithMinimalParameters_CreatesExceptionCorrectly()
    {
        // Arrange & Act
        var exception = new ShapeMismatchException(
            "test_layer",
            OperationType.Conv2D,
            new[] { new long[] { 32, 3, 224, 224 } },
            new[] { new long[] { 32, 64, 224, 224 } },
            "Channel mismatch");

        // Assert
        Assert.NotNull(exception);
        Assert.Equal("test_layer", exception.LayerName);
        Assert.Null(exception.SuggestedFixes);
        Assert.False(exception.BatchSize.HasValue);
    }

    [Fact]
    public void GetDiagnosticReport_GeneratesCorrectFormat()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "encoder.fc2",
            OperationType.MatrixMultiply,
            new[] { new long[] { 32, 256 } },
            new[] { new long[] { 128, 10 } },
            "Dimension 1 mismatch",
            new List<string> { "Suggestion 1", "Suggestion 2" },
            32,
            "encoder.fc1 [32, 256]");

        // Act
        var report = exception.GetDiagnosticReport();

        // Assert
        Assert.NotNull(report);
        Assert.Contains("encoder.fc2", report);
        Assert.Contains("MatrixMultiply", report);
        Assert.Contains("Dimension 1 mismatch", report);
        Assert.Contains("Suggestion 1", report);
        Assert.Contains("Suggestion 2", report);
        Assert.Contains("encoder.fc1", report);
    }

    [Fact]
    public void Exception_Message_IsGeneratedCorrectly()
    {
        // Arrange & Act
        var exception = new ShapeMismatchException(
            "layer1",
            OperationType.Concat,
            new[] { new long[] { 32, 10 } },
            new[] { new long[] { 32, 20 } },
            "Channel mismatch");

        // Assert
        Assert.Contains("layer1", exception.Message);
        Assert.Contains("Concat", exception.Message);
        Assert.Contains("Shape mismatch", exception.Message);
    }

    [Fact]
    public void Constructor_WithAllOptionalParameters_CreatesExceptionCorrectly()
    {
        // Arrange
        var layerName = "fc1";
        var operationType = OperationType.Linear;
        var inputShapes = new[] { new long[] { 32, 784 } };
        var expectedShapes = new[] { new long[] { 784, 256 } };
        var problemDescription = "Weight dimension mismatch";
        var suggestedFixes = new List<string> { "Check weight shape", "Verify input size" };
        var batchSize = 32L;
        var previousLayerContext = "input [32, 784]";

        // Act
        var exception = new ShapeMismatchException(
            layerName,
            operationType,
            inputShapes,
            expectedShapes,
            problemDescription,
            suggestedFixes,
            batchSize,
            previousLayerContext);

        // Assert
        Assert.Equal(layerName, exception.LayerName);
        Assert.Equal(operationType, exception.OperationType);
        Assert.Equal(inputShapes, exception.InputShapes);
        Assert.Equal(expectedShapes, exception.ExpectedShapes);
        Assert.Equal(problemDescription, exception.ProblemDescription);
        Assert.Equal(suggestedFixes, exception.SuggestedFixes);
        Assert.Equal(batchSize, exception.BatchSize);
        Assert.Equal(previousLayerContext, exception.PreviousLayerContext);
    }

    [Fact]
    public void GetDiagnosticReport_WithoutSuggestedFixes_DoesNotIncludeFixesSection()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "layer1",
            OperationType.MatrixMultiply,
            new[] { new long[] { 32, 10 } },
            new[] { new long[] { 10, 5 } },
            "Test error");

        // Act
        var report = exception.GetDiagnosticReport();

        // Assert
        Assert.NotNull(report);
        Assert.DoesNotContain("Suggested fixes", report);
    }

    [Fact]
    public void GetDiagnosticReport_WithBatchSize_IncludesBatchSizeInContext()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "layer1",
            OperationType.Conv2D,
            new[] { new long[] { 32, 3, 224, 224 } },
            new[] { new long[] { 64, 3, 3, 3 } },
            "Channel mismatch",
            batchSize: 32);

        // Act
        var report = exception.GetDiagnosticReport();

        // Assert
        Assert.NotNull(report);
        Assert.Contains("Batch size: 32", report);
    }

    [Fact]
    public void GetDiagnosticReport_WithPreviousLayerContext_IncludesContext()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "layer2",
            OperationType.Linear,
            new[] { new long[] { 32, 256 } },
            new[] { new long[] { 256, 10 } },
            "Dimension mismatch",
            previousLayerContext: "layer1 [32, 256]");

        // Act
        var report = exception.GetDiagnosticReport();

        // Assert
        Assert.NotNull(report);
        Assert.Contains("Previous layer output: layer1 [32, 256]", report);
    }

    [Fact]
    public void GetDiagnosticReport_DifferentOperationTypes_GeneratesCorrectErrorMessages()
    {
        // Arrange & Act
        var matrixMultiplyException = new ShapeMismatchException(
            "fc1", OperationType.MatrixMultiply,
            new[] { new long[] { 32, 10 } },
            new[] { new long[] { 5, 10 } },
            "Test error");
        var conv2DException = new ShapeMismatchException(
            "conv1", OperationType.Conv2D,
            new[] { new long[] { 32, 3, 224, 224 } },
            new[] { new long[] { 64, 3, 3, 3 } },
            "Test error");
        var concatException = new ShapeMismatchException(
            "concat1", OperationType.Concat,
            new[] { new long[] { 32, 10 } },
            new[] { new long[] { 32, 20 } },
            "Test error");

        var matrixMultiplyReport = matrixMultiplyException.GetDiagnosticReport();
        var conv2DReport = conv2DException.GetDiagnosticReport();
        var concatReport = concatException.GetDiagnosticReport();

        // Assert
        Assert.Contains("Matrix multiplication failed", matrixMultiplyReport);
        Assert.Contains("Conv2D operation failed", conv2DReport);
        Assert.Contains("Concatenation failed", concatReport);
    }

    [Fact]
    public void GetDiagnosticReport_MultipleSuggestedFixes_IncludesAllFixes()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "layer1",
            OperationType.MatrixMultiply,
            new[] { new long[] { 32, 256 } },
            new[] { new long[] { 128, 10 } },
            "Dimension mismatch",
            new List<string> { "Fix 1", "Fix 2", "Fix 3" });

        // Act
        var report = exception.GetDiagnosticReport();

        // Assert
        Assert.NotNull(report);
        Assert.Contains("1. Fix 1", report);
        Assert.Contains("2. Fix 2", report);
        Assert.Contains("3. Fix 3", report);
    }
}
