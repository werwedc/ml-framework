using MLFramework.Shapes;

namespace MLFramework.Tests.Shapes;

/// <summary>
/// Unit tests for the ShapeMismatchException class.
/// </summary>
[TestFixture]
public class ShapeMismatchExceptionTests
{
    [Test]
    public void Constructor_WithValidParameters_CreatesExceptionCorrectly()
    {
        // Arrange
        var operationName = "Add";
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("seq", 128))
        };
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 64), new SymbolicDimension("seq", 128))
        };
        var details = "Batch size mismatch";

        // Act
        var exception = new ShapeMismatchException(operationName, expectedShapes, actualShapes, details);

        // Assert
        Assert.That(exception.OperationName, Is.EqualTo(operationName));
        Assert.That(exception.ExpectedShapes.Count, Is.EqualTo(1));
        Assert.That(exception.ActualShapes.Count, Is.EqualTo(1));
        Assert.That(exception.Details, Is.EqualTo(details));
        Assert.That(exception.Message, Does.Contain(operationName));
    }

    [Test]
    public void Constructor_WithEmptyDetails_CreatesExceptionCorrectly()
    {
        // Arrange
        var operationName = "Mul";
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("x", 10))
        };
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("x", 5))
        };

        // Act
        var exception = new ShapeMismatchException(operationName, expectedShapes, actualShapes);

        // Assert
        Assert.That(exception.Details, Is.EqualTo(string.Empty));
        Assert.That(exception.Message, Does.Contain(operationName));
    }

    [Test]
    public void Constructor_WithCustomMessage_CreatesExceptionCorrectly()
    {
        // Arrange
        var operationName = "Reshape";
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("x", 100))
        };
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("x", 10), new SymbolicDimension("y", 5))
        };
        var customMessage = "Custom error message";
        var details = "Total element count mismatch";

        // Act
        var exception = new ShapeMismatchException(
            operationName,
            expectedShapes,
            actualShapes,
            customMessage,
            details);

        // Assert
        Assert.That(exception.Message, Is.EqualTo(customMessage));
        Assert.That(exception.Details, Is.EqualTo(details));
    }

    [Test]
    public void Constructor_WithNullOperationName_ThrowsArgumentNullException()
    {
        // Arrange
        var expectedShapes = new List<SymbolicShape>();
        var actualShapes = new List<SymbolicShape>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
        {
            new ShapeMismatchException(null!, expectedShapes, actualShapes);
        });
    }

    [Test]
    public void Constructor_WithNullExpectedShapes_CreatesExceptionCorrectly()
    {
        // Arrange
        var operationName = "Add";
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("x", 10))
        };

        // Act
        var exception = new ShapeMismatchException(operationName, null!, actualShapes);

        // Assert
        Assert.That(exception.ExpectedShapes, Is.Empty);
        Assert.That(exception.ActualShapes.Count, Is.EqualTo(1));
    }

    [Test]
    public void Constructor_WithNullActualShapes_CreatesExceptionCorrectly()
    {
        // Arrange
        var operationName = "Add";
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("x", 10))
        };

        // Act
        var exception = new ShapeMismatchException(operationName, expectedShapes, null!);

        // Assert
        Assert.That(exception.ExpectedShapes.Count, Is.EqualTo(1));
        Assert.That(exception.ActualShapes, Is.Empty);
    }

    [Test]
    public void Message_WithShapes_IncludesShapeInformation()
    {
        // Arrange
        var operationName = "MatMul";
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(
                new SymbolicDimension("batch", 32),
                new SymbolicDimension("m", 10),
                new SymbolicDimension("n", 20))
        };
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(
                new SymbolicDimension("batch", 32),
                new SymbolicDimension("m", 15),
                new SymbolicDimension("n", 20))
        };

        // Act
        var exception = new ShapeMismatchException(operationName, expectedShapes, actualShapes);

        // Assert
        Assert.That(exception.Message, Does.Contain("Shape mismatch"));
        Assert.That(exception.Message, Does.Contain(operationName));
        Assert.That(exception.Message, Does.Contain("Expected shapes"));
        Assert.That(exception.Message, Does.Contain("Actual shapes"));
    }

    [Test]
    public void ToString_ReturnsFormattedMessage()
    {
        // Arrange
        var operationName = "Conv2D";
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("c", 64))
        };
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("c", 128))
        };

        // Act
        var exception = new ShapeMismatchException(operationName, expectedShapes, actualShapes);

        // Assert
        var result = exception.ToString();
        Assert.That(result, Is.EqualTo(exception.Message));
    }

    [Test]
    public void Properties_PreservedCorrectly()
    {
        // Arrange
        var operationName = "Transpose";
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(
                new SymbolicDimension("h", 256),
                new SymbolicDimension("w", 256))
        };
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(
                new SymbolicDimension("h", 256),
                new SymbolicDimension("w", 256),
                new SymbolicDimension("c", 3))
        };
        var details = "Unexpected third dimension";

        // Act
        var exception = new ShapeMismatchException(operationName, expectedShapes, actualShapes, details);

        // Assert
        Assert.That(exception.OperationName, Is.EqualTo(operationName));
        Assert.That(exception.ExpectedShapes.Count, Is.EqualTo(1));
        Assert.That(exception.ExpectedShapes[0].Rank, Is.EqualTo(2));
        Assert.That(exception.ActualShapes.Count, Is.EqualTo(1));
        Assert.That(exception.ActualShapes[0].Rank, Is.EqualTo(3));
        Assert.That(exception.Details, Is.EqualTo(details));
    }

    [Test]
    public void Constructor_WithMultipleShapes_HandlesCorrectly()
    {
        // Arrange
        var operationName = "Concat";
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

        // Act
        var exception = new ShapeMismatchException(operationName, expectedShapes, actualShapes);

        // Assert
        Assert.That(exception.ExpectedShapes.Count, Is.EqualTo(2));
        Assert.That(exception.ActualShapes.Count, Is.EqualTo(2));
    }

    [Test]
    public void Constructor_WithSymbolicDimensions_HandlesCorrectly()
    {
        // Arrange
        var operationName = "DynamicOp";
        var expectedShapes = new List<SymbolicShape>
        {
            new SymbolicShape(
                new SymbolicDimension("batch", null, 1, 256),
                new SymbolicDimension("seq", null, 1, 1024))
        };
        var actualShapes = new List<SymbolicShape>
        {
            new SymbolicShape(
                new SymbolicDimension("batch", 32),
                new SymbolicDimension("seq", 512))
        };

        // Act
        var exception = new ShapeMismatchException(operationName, expectedShapes, actualShapes);

        // Assert
        Assert.That(exception.ExpectedShapes[0].GetDimension(0).IsKnown(), Is.False);
        Assert.That(exception.ActualShapes[0].GetDimension(0).IsKnown(), Is.True);
    }
}
