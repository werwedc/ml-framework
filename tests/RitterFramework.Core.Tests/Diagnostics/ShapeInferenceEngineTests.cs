using RitterFramework.Core;
using RitterFramework.Core.Diagnostics;
using Xunit;

namespace MLFramework.Tests.Diagnostics;

/// <summary>
/// Unit tests for ShapeInferenceEngine class.
/// </summary>
public class ShapeInferenceEngineTests
{
    private IShapeInferenceEngine _engine;

    public ShapeInferenceEngineTests()
    {
        var registry = new DefaultOperationMetadataRegistry();
        _engine = new DefaultShapeInferenceEngine(registry);
    }

    [Fact]
    public void InferOutputShape_MatrixMultiply2D_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 }, new long[] { 10, 5 } };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.MatrixMultiply,
            inputShapes);

        // Assert
        Assert.Equal(2, outputShape.Length);
        Assert.Equal(32, outputShape[0]);
        Assert.Equal(5, outputShape[1]);
    }

    [Fact]
    public void InferOutputShape_MatrixMultiply3D_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 4, 32, 10 }, new long[] { 10, 5 } };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.MatrixMultiply,
            inputShapes);

        // Assert
        Assert.Equal(3, outputShape.Length);
        Assert.Equal(4, outputShape[0]);
        Assert.Equal(32, outputShape[1]);
        Assert.Equal(5, outputShape[2]);
    }

    [Fact]
    public void InferOutputShape_Conv2D_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 3, 224, 224 }, new long[] { 64, 3, 3, 3 } };
        var parameters = new Dictionary<string, object>
        {
            { "stride", new[] { 1, 1 } },
            { "padding", new[] { 0, 0 } }
        };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.Conv2D,
            inputShapes,
            parameters);

        // Assert
        Assert.Equal(4, outputShape.Length);
        Assert.Equal(32, outputShape[0]); // batch
        Assert.Equal(64, outputShape[1]); // output channels
        Assert.Equal(222, outputShape[2]); // (224 - 3 + 0) / 1 + 1
        Assert.Equal(222, outputShape[3]);
    }

    [Fact]
    public void InferOutputShape_Conv2DWithPadding_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 3, 224, 224 }, new long[] { 64, 3, 3, 3 } };
        var parameters = new Dictionary<string, object>
        {
            { "stride", new[] { 1, 1 } },
            { "padding", new[] { 1, 1 } }
        };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.Conv2D,
            inputShapes,
            parameters);

        // Assert
        Assert.Equal(224, outputShape[2]); // Same padding maintains size
        Assert.Equal(224, outputShape[3]);
    }

    [Fact]
    public void InferOutputShape_Conv2DWithStride_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 3, 224, 224 }, new long[] { 64, 3, 3, 3 } };
        var parameters = new Dictionary<string, object>
        {
            { "stride", new[] { 2, 2 } },
            { "padding", new[] { 0, 0 } }
        };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.Conv2D,
            inputShapes,
            parameters);

        // Assert
        Assert.Equal(111, outputShape[2]); // (224 - 3 + 0) / 2 + 1
        Assert.Equal(111, outputShape[3]);
    }

    [Fact]
    public void InferOutputShape_Concat_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 }, new long[] { 32, 20 }, new long[] { 32, 15 } };
        var parameters = new Dictionary<string, object>
        {
            { "axis", 1 }
        };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.Concat,
            inputShapes,
            parameters);

        // Assert
        Assert.Equal(2, outputShape.Length);
        Assert.Equal(32, outputShape[0]);
        Assert.Equal(45, outputShape[1]); // 10 + 20 + 15
    }

    [Fact]
    public void InferOutputShape_Stack_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 }, new long[] { 32, 10 }, new long[] { 32, 10 } };
        var parameters = new Dictionary<string, object>
        {
            { "axis", 0 }
        };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.Stack,
            inputShapes,
            parameters);

        // Assert
        Assert.Equal(3, outputShape.Length);
        Assert.Equal(3, outputShape[0]); // Number of tensors
        Assert.Equal(32, outputShape[1]);
        Assert.Equal(10, outputShape[2]);
    }

    [Fact]
    public void InferOutputShape_Transpose2D_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 } };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.Transpose,
            inputShapes);

        // Assert
        Assert.Equal(2, outputShape.Length);
        Assert.Equal(10, outputShape[0]);
        Assert.Equal(32, outputShape[1]);
    }

    [Fact]
    public void InferOutputShape_Reshape_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 } };
        var parameters = new Dictionary<string, object>
        {
            { "shape", new long[] { 320 } }
        };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.Reshape,
            inputShapes,
            parameters);

        // Assert
        Assert.Single(outputShape);
        Assert.Equal(320, outputShape[0]);
    }

    [Fact]
    public void InferOutputShape_ReshapeWithInferredDimension_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 } };
        var parameters = new Dictionary<string, object>
        {
            { "shape", new long[] { -1, 8 } }
        };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.Reshape,
            inputShapes,
            parameters);

        // Assert
        Assert.Equal(2, outputShape.Length);
        Assert.Equal(40, outputShape[0]); // 320 / 8
        Assert.Equal(8, outputShape[1]);
    }

    [Fact]
    public void InferOutputShape_Linear_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 784 }, new long[] { 10, 784 } };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.Linear,
            inputShapes);

        // Assert
        Assert.Equal(2, outputShape.Length);
        Assert.Equal(32, outputShape[0]);
        Assert.Equal(10, outputShape[1]);
    }

    [Fact]
    public void InferOutputShape_Broadcast_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 1 }, new long[] { 32, 10 } };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.Broadcast,
            inputShapes);

        // Assert
        Assert.Equal(2, outputShape.Length);
        Assert.Equal(32, outputShape[0]);
        Assert.Equal(10, outputShape[1]);
    }

    [Fact]
    public void InferOutputShape_UnknownOperation_ReturnsFirstInputShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 } };

        // Act
        var outputShape = _engine.InferOutputShape(
            (OperationType)999,
            inputShapes);

        // Assert
        Assert.Equal(inputShapes[0], outputShape);
    }

    [Fact]
    public void InferOutputShape_ReshapeWithMultipleInferredDimensions_ThrowsException()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 } };
        var parameters = new Dictionary<string, object>
        {
            { "shape", new long[] { -1, -1 } }
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
        {
            _engine.InferOutputShape(
                OperationType.Reshape,
                inputShapes,
                parameters);
        });
    }

    [Fact]
    public void InferOutputShape_BroadcastIncompatibleShapes_ReturnsPartialResult()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 }, new long[] { 20, 10 } }; // Incompatible batch sizes

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.Broadcast,
            inputShapes);

        // Assert - Should return partial result (not throw)
        Assert.NotNull(outputShape);
        Assert.Equal(2, outputShape.Length);
    }

    [Fact]
    public void InferOutputShape_ConcatWithoutAxisParameter_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 }, new long[] { 32, 20 } };

        // Act - Default axis is 0
        var outputShape = _engine.InferOutputShape(
            OperationType.Concat,
            inputShapes);

        // Assert
        Assert.Equal(2, outputShape.Length);
        Assert.Equal(64, outputShape[0]); // 32 + 32
        Assert.Equal(10, outputShape[1]);
    }

    [Fact]
    public void InferOutputShape_StackWithoutAxisParameter_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 10 }, new long[] { 32, 10 } };

        // Act - Default axis is 0
        var outputShape = _engine.InferOutputShape(
            OperationType.Stack,
            inputShapes);

        // Assert
        Assert.Equal(3, outputShape.Length);
        Assert.Equal(2, outputShape[0]); // Number of tensors
        Assert.Equal(32, outputShape[1]);
        Assert.Equal(10, outputShape[2]);
    }
}
