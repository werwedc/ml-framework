using NUnit.Framework;

namespace RitterFramework.Core.Tests.Exceptions;

/// <summary>
/// Unit tests for the ShapeMismatchException class.
/// </summary>
[TestFixture]
public class ShapeMismatchExceptionTests
{
    [Test]
    public void DefaultConstructor_CreatesEmptyException()
    {
        // Arrange & Act
        var exception = new ShapeMismatchException();

        // Assert
        Assert.That(exception.Message, Is.Empty);
        Assert.That(exception.LayerName, Is.Null);
        Assert.That(exception.InputShapes, Is.Empty);
        Assert.That(exception.ExpectedShapes, Is.Empty);
        Assert.That(exception.SuggestedFixes, Is.Empty);
    }

    [Test]
    public void MessageConstructor_CreatesExceptionWithMessage()
    {
        // Arrange
        var message = "Shape mismatch occurred";

        // Act
        var exception = new ShapeMismatchException(message);

        // Assert
        Assert.That(exception.Message, Is.EqualTo(message));
    }

    [Test]
    public void MessageAndInnerExceptionConstructor_CreatesExceptionWithBoth()
    {
        // Arrange
        var message = "Shape mismatch occurred";
        var innerException = new InvalidOperationException("Inner error");

        // Act
        var exception = new ShapeMismatchException(message, innerException);

        // Assert
        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.InnerException, Is.SameAs(innerException));
    }

    [Test]
    public void FullConstructor_CreatesExceptionWithAllProperties()
    {
        // Arrange
        var message = "Matrix multiplication failed";
        var layerName = "encoder.fc2";
        var operationType = OperationType.MatrixMultiply;
        var inputShapes = new List<long[]> { new[] { 32L, 256L } };
        var expectedShapes = new List<long[]> { new[] { -1L, 128L } };
        var problemDescription = "Dimension 1 of input (256) does not match dimension 0 of weight (128)";
        var suggestedFixes = new List<string>
        {
            "Check layer configuration",
            "Verify input features match expected dimension"
        };

        // Act
        var exception = new ShapeMismatchException(
            message,
            layerName,
            operationType,
            inputShapes,
            expectedShapes,
            problemDescription,
            suggestedFixes);

        // Assert
        Assert.That(exception.Message, Is.EqualTo(message));
        Assert.That(exception.LayerName, Is.EqualTo(layerName));
        Assert.That(exception.OperationType, Is.EqualTo(operationType));
        Assert.That(exception.InputShapes.Count, Is.EqualTo(1));
        Assert.That(exception.InputShapes[0], Is.EqualTo(new[] { 32L, 256L }));
        Assert.That(exception.ExpectedShapes.Count, Is.EqualTo(1));
        Assert.That(exception.ExpectedShapes[0], Is.EqualTo(new[] { -1L, 128L }));
        Assert.That(exception.ProblemDescription, Is.EqualTo(problemDescription));
        Assert.That(exception.SuggestedFixes.Count, Is.EqualTo(2));
        Assert.That(exception.SuggestedFixes[0], Is.EqualTo("Check layer configuration"));
    }

    [Test]
    public void FullConstructor_WithNullSuggestedFixes_CreatesEmptyList()
    {
        // Arrange
        var message = "Shape mismatch";
        var inputShapes = new List<long[]> { new[] { 32L, 256L } };
        var expectedShapes = new List<long[]> { new[] { -1L, 128L } };

        // Act
        var exception = new ShapeMismatchException(
            message,
            null,
            OperationType.Linear,
            inputShapes,
            expectedShapes,
            null,
            null);

        // Assert
        Assert.That(exception.SuggestedFixes, Is.Not.Null);
        Assert.That(exception.SuggestedFixes, Is.Empty);
    }

    [Test]
    public void GetDiagnosticReport_WithAllFields_ReturnsFormattedReport()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "Matrix multiplication failed",
            "encoder.fc2",
            OperationType.MatrixMultiply,
            new List<long[]> { new[] { 32L, 256L } },
            new List<long[]> { new[] { -1L, 128L } },
            "Dimension 1 of input (256) does not match dimension 0 of weight (128)",
            new List<string> { "Check layer configuration", "Verify input features match expected dimension" });

        // Act
        var report = exception.GetDiagnosticReport();

        // Assert
        Assert.That(report, Does.Contain("MLFramework.ShapeMismatchException"));
        Assert.That(report, Does.Contain("encoder.fc2"));
        Assert.That(report, Does.Contain("MatrixMultiply"));
        Assert.That(report, Does.Contain("[32, 256]"));
        Assert.That(report, Does.Contain("[*, 128]"));
        Assert.That(report, Does.Contain("Dimension 1 of input (256) does not match dimension 0 of weight (128)"));
        Assert.That(report, Does.Contain("Suggested fixes"));
        Assert.That(report, Does.Contain("1. Check layer configuration"));
        Assert.That(report, Does.Contain("2. Verify input features match expected dimension"));
    }

    [Test]
    public void GetDiagnosticReport_WithMultipleShapes_CorrectlyLabelsThem()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "Concatenation failed",
            "concat_layer",
            OperationType.Concat,
            new List<long[]>
            {
                new[] { 32L, 128L },
                new[] { 32L, 256L }
            },
            new List<long[]>
            {
                new[] { 32L, 128L },
                new[] { 32L, 128L }
            },
            "Shapes don't match on axis 1");

        // Act
        var report = exception.GetDiagnosticReport();

        // Assert
        Assert.That(report, Does.Contain("Input shape [0]"));
        Assert.That(report, Does.Contain("Input shape [1]"));
        Assert.That(report, Does.Contain("[32, 128]"));
        Assert.That(report, Does.Contain("[32, 256]"));
    }

    [Test]
    public void GetDiagnosticReport_WithoutLayerName_StillShowsShapes()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "Shape mismatch",
            null,
            OperationType.Linear,
            new List<long[]> { new[] { 32L, 256L } },
            new List<long[]> { new[] { 32L, 128L } });

        // Act
        var report = exception.GetDiagnosticReport();

        // Assert
        Assert.That(report, Does.Contain("[32, 256]"));
        Assert.That(report, Does.Contain("[32, 128]"));
        Assert.That(report, Does.Not.Contain("Layer:"));
    }

    [Test]
    public void GetDiagnosticReport_WithoutSuggestedFixes_DoesNotShowSection()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "Shape mismatch",
            "test_layer",
            OperationType.Reshape,
            new List<long[]> { new[] { 784L } },
            new List<long[]> { new[] { 28L, 28L } });

        // Act
        var report = exception.GetDiagnosticReport();

        // Assert
        Assert.That(report, Does.Not.Contain("Suggested fixes"));
    }

    [Test]
    public void GetDiagnosticReport_WithBroadcastingFailure_FormatCorrectly()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "Broadcasting failed",
            "add_op",
            OperationType.Broadcast,
            new List<long[]>
            {
                new[] { 32L, 10L },
                new[] { 20L, 10L }
            },
            new List<long[]>
            {
                new[] { 32L, 10L },
                new[] { 32L, 10L }
            },
            "Cannot broadcast shapes [32, 10] and [20, 10]");

        // Act
        var report = exception.GetDiagnosticReport();

        // Assert
        Assert.That(report, Does.Contain("Broadcasting failed"));
        Assert.That(report, Does.Contain("[32, 10]"));
        Assert.That(report, Does.Contain("[20, 10]"));
        Assert.That(report, Does.Contain("Cannot broadcast shapes"));
    }

    [Test]
    public void Properties_AreMutable()
    {
        // Arrange
        var exception = new ShapeMismatchException("Test");

        // Act
        exception.LayerName = "new_layer";
        exception.OperationType = OperationType.Conv2D;
        exception.ProblemDescription = "New problem";
        exception.SuggestedFixes = new List<string> { "Fix it" };

        // Assert
        Assert.That(exception.LayerName, Is.EqualTo("new_layer"));
        Assert.That(exception.OperationType, Is.EqualTo(OperationType.Conv2D));
        Assert.That(exception.ProblemDescription, Is.EqualTo("New problem"));
        Assert.That(exception.SuggestedFixes[0], Is.EqualTo("Fix it"));
    }

    [Test]
    public void Properties_CanAddShapes()
    {
        // Arrange
        var exception = new ShapeMismatchException("Test");

        // Act
        exception.InputShapes.Add(new[] { 10L, 20L });
        exception.ExpectedShapes.Add(new[] { 10L, 30L });

        // Assert
        Assert.That(exception.InputShapes.Count, Is.EqualTo(1));
        Assert.That(exception.ExpectedShapes.Count, Is.EqualTo(1));
        Assert.That(exception.InputShapes[0], Is.EqualTo(new[] { 10L, 20L }));
        Assert.That(exception.ExpectedShapes[0], Is.EqualTo(new[] { 10L, 30L }));
    }

    [Test]
    public void GetDiagnosticReport_OutputFormatMatchesSpecExample()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "Matrix multiplication failed in layer 'encoder.fc2'",
            "encoder.fc2",
            OperationType.MatrixMultiply,
            new List<long[]> { new[] { 32L, 256L } },
            new List<long[]> { new[] { -1L, 128L } },
            "Dimension 1 of input (256) does not match dimension 0 of weight (128)",
            new List<string>
            {
                "Check layer configuration",
                "Verify input features match expected dimension"
            });

        // Act
        var report = exception.GetDiagnosticReport();

        // Assert - Verify format matches spec example structure
        var lines = report.Split('\n');
        Assert.That(lines.Length, Is.GreaterThan(5));
        Assert.That(lines[0], Does.Contain("MLFramework.ShapeMismatchException"));
    }
}
