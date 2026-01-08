using MLFramework.Shapes;

namespace MLFramework.Tests.Shapes;

/// <summary>
/// Unit tests for the ShapeErrorReporter class.
/// </summary>
[TestFixture]
public class ShapeErrorReporterTests
{
    private ShapeErrorReporter _reporter = null!;

    [SetUp]
    public void SetUp()
    {
        _reporter = new ShapeErrorReporter();
    }

    [Test]
    public void FormatError_WithValidException_ReturnsFormattedMessage()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "MatMul",
            new List<SymbolicShape>
            {
                new SymbolicShape(new SymbolicDimension("m", 10), new SymbolicDimension("n", 20))
            },
            new List<SymbolicShape>
            {
                new SymbolicShape(new SymbolicDimension("m", 15), new SymbolicDimension("n", 20))
            },
            "Inner dimension mismatch");

        // Act
        var result = _reporter.FormatError(exception);

        // Assert
        Assert.That(result, Does.Contain("Shape Mismatch Error"));
        Assert.That(result, Does.Contain("MatMul"));
        Assert.That(result, Does.Contain("Expected Shapes"));
        Assert.That(result, Does.Contain("Actual Shapes"));
        Assert.That(result, Does.Contain("Details"));
        Assert.That(result, Does.Contain("Inner dimension mismatch"));
    }

    [Test]
    public void FormatError_WithNullException_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => _reporter.FormatError(null!));
    }

    [Test]
    public void FormatError_WithEmptyShapes_ReturnsCorrectFormat()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "TestOp",
            new List<SymbolicShape>(),
            new List<SymbolicShape>(),
            "No shapes available");

        // Act
        var result = _reporter.FormatError(exception);

        // Assert
        Assert.That(result, Does.Contain("TestOp"));
        Assert.That(result, Does.Contain("No shapes available"));
    }

    [Test]
    public void SuggestFix_WithMatMulException_ReturnsMatMulSuggestion()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "MatMul",
            new List<SymbolicShape>(),
            new List<SymbolicShape>(),
            "Dimension mismatch");

        // Act
        var suggestion = _reporter.SuggestFix(exception);

        // Assert
        Assert.That(suggestion, Does.Contain("inner dimensions"));
        Assert.That(suggestion, Does.Contain("matrix multiplication"));
    }

    [Test]
    public void SuggestFix_WithReshapeException_ReturnsReshapeSuggestion()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "Reshape",
            new List<SymbolicShape>(),
            new List<SymbolicShape>(),
            "Element count mismatch");

        // Act
        var suggestion = _reporter.SuggestFix(exception);

        // Assert
        Assert.That(suggestion, Does.Contain("reshape"));
        Assert.That(suggestion, Does.Contain("-1"));
    }

    [Test]
    public void SuggestFix_WithBroadcastException_ReturnsBroadcastSuggestion()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "Broadcast",
            new List<SymbolicShape>(),
            new List<SymbolicShape>(),
            "Shapes not compatible");

        // Act
        var suggestion = _reporter.SuggestFix(exception);

        // Assert
        Assert.That(suggestion, Does.Contain("broadcasting"));
        Assert.That(suggestion, Does.Contain(".unsqueeze()") || suggestion.Contains(".expand()"));
    }

    [Test]
    public void SuggestFix_WithSequenceException_ReturnsSequenceSuggestion()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "Sequence",
            new List<SymbolicShape>(),
            new List<SymbolicShape>(),
            "Multiple errors");

        // Act
        var suggestion = _reporter.SuggestFix(exception);

        // Assert
        Assert.That(suggestion, Does.Contain("sequence"));
        Assert.That(suggestion, Does.Contain("output tensor names"));
    }

    [Test]
    public void SuggestFix_WithRankMismatch_ReturnsRankSuggestion()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "Add",
            new List<SymbolicShape>
            {
                new SymbolicShape(new SymbolicDimension("x", 10), new SymbolicDimension("y", 20))
            },
            new List<SymbolicShape>
            {
                new SymbolicShape(new SymbolicDimension("x", 10))
            },
            "Rank mismatch");

        // Act
        var suggestion = _reporter.SuggestFix(exception);

        // Assert
        Assert.That(suggestion, Does.Contain("Rank mismatch"));
        Assert.That(suggestion, Does.Contain(".unsqueeze()") || suggestion.Contains(".squeeze()"));
    }

    [Test]
    public void SuggestFix_WithDimensionOne_ReturnsRepeatOrExpandSuggestion()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "Add",
            new List<SymbolicShape>
            {
                new SymbolicShape(new SymbolicDimension("x", 10), new SymbolicDimension("y", 20))
            },
            new List<SymbolicShape>
            {
                new SymbolicShape(new SymbolicDimension("x", 1), new SymbolicDimension("y", 20))
            },
            "Dimension 0 is 1 but should be 10");

        // Act
        var suggestion = _reporter.SuggestFix(exception);

        // Assert
        Assert.That(suggestion, Does.Contain("1") && suggestion.Contains("10"));
        Assert.That(suggestion, Does.Contain(".repeat()") || suggestion.Contains(".expand()"));
    }

    [Test]
    public void SuggestFix_WithGeneralMismatch_ReturnsGeneralSuggestion()
    {
        // Arrange
        var exception = new ShapeMismatchException(
            "Add",
            new List<SymbolicShape>(),
            new List<SymbolicShape>(),
            "Some error");

        // Act
        var suggestion = _reporter.SuggestFix(exception);

        // Assert
        Assert.That(suggestion, Is.Not.Null.Or.Empty);
    }

    [Test]
    public void SuggestFix_WithNullException_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => _reporter.SuggestFix(null!));
    }

    [Test]
    public void VisualizeShapes_WithSingleShape_ReturnsVisualization()
    {
        // Arrange
        var shapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64))
        };

        // Act
        var result = _reporter.VisualizeShapes(shapes);

        // Assert
        Assert.That(result, Does.Contain("Shape Visualization"));
        Assert.That(result, Does.Contain("Shape 1"));
        Assert.That(result, Does.Contain("batch=32"));
        Assert.That(result, Does.Contain("feat=64"));
    }

    [Test]
    public void VisualizeShapes_WithMultipleShapes_ReturnsVisualization()
    {
        // Arrange
        var shapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64)),
            new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 128))
        };

        // Act
        var result = _reporter.VisualizeShapes(shapes);

        // Assert
        Assert.That(result, Does.Contain("Shape 1"));
        Assert.That(result, Does.Contain("Shape 2"));
    }

    [Test]
    public void VisualizeShapes_WithNullShapes_ReturnsNoShapesMessage()
    {
        // Act
        var result = _reporter.VisualizeShapes(null!);

        // Assert
        Assert.That(result, Does.Contain("No shapes to visualize"));
    }

    [Test]
    public void VisualizeShapes_WithEmptyShapes_ReturnsNoShapesMessage()
    {
        // Arrange
        var shapes = new List<SymbolicShape>();

        // Act
        var result = _reporter.VisualizeShapes(shapes);

        // Assert
        Assert.That(result, Does.Contain("No shapes to visualize"));
    }

    [Test]
    public void VisualizeShapes_With2DShape_Returns2DVisualization()
    {
        // Arrange
        var shapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("rows", 3), new SymbolicDimension("cols", 4))
        };

        // Act
        var result = _reporter.VisualizeShapes(shapes);

        // Assert
        Assert.That(result, Does.Contain("2D Tensor"));
    }

    [Test]
    public void VisualizeShapes_With1DShape_Returns1DVisualization()
    {
        // Arrange
        var shapes = new List<SymbolicShape>
        {
            new SymbolicShape(new SymbolicDimension("len", 10))
        };

        // Act
        var result = _reporter.VisualizeShapes(shapes);

        // Assert
        Assert.That(result, Does.Contain("1D Tensor"));
    }

    [Test]
    public void VisualizeShapes_With3DShape_Returns3DVisualization()
    {
        // Arrange
        var shapes = new List<SymbolicShape>
        {
            new SymbolicShape(
                new SymbolicDimension("batch", 32),
                new SymbolicDimension("height", 224),
                new SymbolicDimension("width", 224))
        };

        // Act
        var result = _reporter.VisualizeShapes(shapes);

        // Assert
        Assert.That(result, Does.Contain("3D Tensor"));
    }

    [Test]
    public void VisualizeShapes_WithSymbolicDimensions_ReturnsVisualizationWithBounds()
    {
        // Arrange
        var shapes = new List<SymbolicShape>
        {
            new SymbolicShape(
                new SymbolicDimension("batch", null, 1, 256),
                new SymbolicDimension("seq", null, 1, 1024))
        };

        // Act
        var result = _reporter.VisualizeShapes(shapes);

        // Assert
        Assert.That(result, Does.Contain("batch[1..256]"));
        Assert.That(result, Does.Contain("seq[1..1024]"));
    }

    [Test]
    public void CompareShapes_WithMatchingShapes_ReturnsAllMatches()
    {
        // Arrange
        var shapeA = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64));
        var shapeB = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64));

        // Act
        var result = _reporter.CompareShapes(shapeA, shapeB);

        // Assert
        Assert.That(result, Does.Contain("Shape Comparison"));
        Assert.That(result, Does.Contain("Match?"));
        Assert.That(result, Does.Contain("✓"));
    }

    [Test]
    public void CompareShapes_WithMismatchedShapes_ReturnsMismatches()
    {
        // Arrange
        var shapeA = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64));
        var shapeB = new SymbolicShape(new SymbolicDimension("batch", 64), new SymbolicDimension("feat", 64));

        // Act
        var result = _reporter.CompareShapes(shapeA, shapeB);

        // Assert
        Assert.That(result, Does.Contain("✗"));
        Assert.That(result, Does.Contain("batch=32"));
        Assert.That(result, Does.Contain("batch=64"));
    }

    [Test]
    public void CompareShapes_WithDifferentRanks_ReturnsCorrectComparison()
    {
        // Arrange
        var shapeA = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64));
        var shapeB = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64), new SymbolicDimension("seq", 128));

        // Act
        var result = _reporter.CompareShapes(shapeA, shapeB);

        // Assert
        Assert.That(result, Does.Contain("N/A"));
    }

    [Test]
    public void CompareShapes_WithSymbolicDimensions_ReturnsPartialMatches()
    {
        // Arrange
        var shapeA = new SymbolicShape(
            new SymbolicDimension("batch", null, 1, 256),
            new SymbolicDimension("feat", 64));
        var shapeB = new SymbolicShape(
            new SymbolicDimension("batch", 32),
            new SymbolicDimension("feat", 64));

        // Act
        var result = _reporter.CompareShapes(shapeA, shapeB);

        // Assert
        Assert.That(result, Does.Contain("~")); // Partial match indicator
    }

    [Test]
    public void CompareShapes_WithNullShapes_ThrowsArgumentNullException()
    {
        var shape = new SymbolicShape(new SymbolicDimension("x", 10));

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => _reporter.CompareShapes(null!, shape));
        Assert.Throws<ArgumentNullException>(() => _reporter.CompareShapes(shape, null!));
    }

    [Test]
    public void CompareShapes_WithBroadcastCompatibleShapes_ReturnsMatches()
    {
        // Arrange
        var shapeA = new SymbolicShape(new SymbolicDimension("batch", 32), new SymbolicDimension("feat", 64));
        var shapeB = new SymbolicShape(new SymbolicDimension("feat", 64));

        // Act
        var result = _reporter.CompareShapes(shapeA, shapeB);

        // Assert
        Assert.That(result, Does.Contain("✓")); // Broadcast-compatible
    }
}
