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

    #region Graph Shape Inference Tests

    [Fact]
    public void InferGraphShapes_SimpleLinearModel_CorrectShapes()
    {
        // Arrange
        var graph = new ComputationGraph();

        // Create nodes: input -> linear1 -> linear2 -> output
        graph.AddNode(new OperationNode("input", OperationType.Linear));
        graph.AddNode(new OperationNode("linear1", OperationType.Linear, new[] { "input" }));
        graph.AddNode(new OperationNode("linear2", OperationType.Linear, new[] { "linear1" }));
        graph.AddNode(new OperationNode("output", OperationType.Linear, new[] { "linear2" }));

        // Add edges
        graph.AddEdge("input", "linear1");
        graph.AddEdge("linear1", "linear2");
        graph.AddEdge("linear2", "output");

        var inputShapes = new Dictionary<string, long[]>
        {
            { "input", new long[] { 32, 128 } }
        };

        // Act
        var result = _engine.InferGraphShapes(graph, inputShapes);

        // Assert
        Assert.Equal(4, result.Count);
        Assert.True(result.ContainsKey("input"));
        Assert.True(result.ContainsKey("linear1"));
        Assert.True(result.ContainsKey("linear2"));
        Assert.True(result.ContainsKey("output"));
    }

    [Fact]
    public void InferGraphShapes_CNNModel_CorrectShapes()
    {
        // Arrange
        var graph = new ComputationGraph();

        // Create nodes: input -> conv1 -> pool1 -> conv2 -> output
        graph.AddNode(new OperationNode("input", OperationType.Conv2D));
        graph.AddNode(new OperationNode("conv1", OperationType.Conv2D, new[] { "input" }));
        graph.AddNode(new OperationNode("pool1", OperationType.MaxPool2D, new[] { "conv1" }));
        graph.AddNode(new OperationNode("conv2", OperationType.Conv2D, new[] { "pool1" }));

        // Add edges
        graph.AddEdge("input", "conv1");
        graph.AddEdge("conv1", "pool1");
        graph.AddEdge("pool1", "conv2");

        var inputShapes = new Dictionary<string, long[]>
        {
            { "input", new long[] { 32, 3, 224, 224 } }
        };

        // Act
        var result = _engine.InferGraphShapes(graph, inputShapes);

        // Assert
        Assert.Equal(4, result.Count);
        Assert.True(result.ContainsKey("input"));
        Assert.True(result.ContainsKey("conv1"));
        Assert.True(result.ContainsKey("pool1"));
        Assert.True(result.ContainsKey("conv2"));
    }

    [Fact]
    public void InferGraphShapes_WithCycle_ThrowsException()
    {
        // Arrange
        var graph = new ComputationGraph();

        // Create nodes with cycle: a -> b -> c -> a
        graph.AddNode(new OperationNode("a", OperationType.Linear));
        graph.AddNode(new OperationNode("b", OperationType.Linear, new[] { "a" }));
        graph.AddNode(new OperationNode("c", OperationType.Linear, new[] { "b" }));

        // Add edges to create cycle
        graph.AddEdge("a", "b");
        graph.AddEdge("b", "c");
        graph.AddEdge("c", "a");

        var inputShapes = new Dictionary<string, long[]>
        {
            { "a", new long[] { 32, 128 } }
        };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            _engine.InferGraphShapes(graph, inputShapes));
    }

    [Fact]
    public void InferGraphShapes_BranchingGraph_CorrectShapes()
    {
        // Arrange
        var graph = new ComputationGraph();

        // Create nodes: input -> (a, b) -> concat -> output
        graph.AddNode(new OperationNode("input", OperationType.Linear));
        graph.AddNode(new OperationNode("a", OperationType.Linear, new[] { "input" }));
        graph.AddNode(new OperationNode("b", OperationType.Linear, new[] { "input" }));
        graph.AddNode(new OperationNode("concat", OperationType.Concat, new[] { "a", "b" }));

        // Add edges
        graph.AddEdge("input", "a");
        graph.AddEdge("input", "b");
        graph.AddEdge("a", "concat");
        graph.AddEdge("b", "concat");

        var inputShapes = new Dictionary<string, long[]>
        {
            { "input", new long[] { 32, 128 } }
        };

        // Act
        var result = _engine.InferGraphShapes(graph, inputShapes);

        // Assert
        Assert.Equal(4, result.Count);
        Assert.True(result.ContainsKey("input"));
        Assert.True(result.ContainsKey("a"));
        Assert.True(result.ContainsKey("b"));
        Assert.True(result.ContainsKey("concat"));
    }

    #endregion

    #region Validation Tests

    [Fact]
    public void ValidateOperation_ValidShapes_ReturnsTrue()
    {
        // Arrange
        var shapes = new[] { new long[] { 32, 128 }, new long[] { 128, 64 } };

        // Act
        var result = _engine.ValidateOperation(OperationType.MatrixMultiply, shapes, null, out string errorMessage);

        // Assert
        Assert.True(result);
        Assert.Null(errorMessage);
    }

    [Fact]
    public void ValidateOperation_InvalidShapes_ReturnsFalse()
    {
        // Arrange
        var shapes = new[] { new long[] { 32, 128 }, new long[] { 256, 64 } };

        // Act
        var result = _engine.ValidateOperation(OperationType.MatrixMultiply, shapes, null, out string errorMessage);

        // Assert
        Assert.False(result);
        Assert.NotNull(errorMessage);
    }

    [Fact]
    public void ValidateOperation_NoInputShapes_ReturnsFalse()
    {
        // Arrange
        var shapes = Array.Empty<long[]>();

        // Act
        var result = _engine.ValidateOperation(OperationType.MatrixMultiply, shapes, null, out string errorMessage);

        // Assert
        Assert.False(result);
        Assert.NotNull(errorMessage);
        Assert.Contains("No input shapes", errorMessage);
    }

    #endregion

    #region Conv1D Tests

    [Fact]
    public void InferOutputShape_Conv1D_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 3, 100 }, new long[] { 64, 3, 5 } };
        var parameters = new Dictionary<string, object>
        {
            { "stride", 1 },
            { "padding", 0 }
        };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.Conv1D,
            inputShapes,
            parameters);

        // Assert
        Assert.Equal(3, outputShape.Length);
        Assert.Equal(32, outputShape[0]); // batch
        Assert.Equal(64, outputShape[1]); // output channels
        Assert.Equal(96, outputShape[2]); // (100 + 2*0 - 5) / 1 + 1
    }

    [Fact]
    public void InferOutputShape_Conv1DWithPadding_ReturnsCorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 3, 100 }, new long[] { 64, 3, 5 } };
        var parameters = new Dictionary<string, object>
        {
            { "stride", 1 },
            { "padding", 2 }
        };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.Conv1D,
            inputShapes,
            parameters);

        // Assert
        Assert.Equal(100, outputShape[2]); // Same padding maintains size
    }

    #endregion

    #region Flatten Tests

    [Fact]
    public void InferOutputShape_Flatten_CorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 3, 224, 224 } };
        var parameters = new Dictionary<string, object>
        {
            { "start_dim", 1 },
            { "end_dim", 3 }
        };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.Flatten,
            inputShapes,
            parameters);

        // Assert
        Assert.Equal(2, outputShape.Length);
        Assert.Equal(32, outputShape[0]);
        Assert.Equal(3 * 224 * 224, outputShape[1]);
    }

    [Fact]
    public void InferOutputShape_FlattenDefaultDimensions_CorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 3, 224, 224 } };

        // Act - Default: start_dim=1, end_dim=last
        var outputShape = _engine.InferOutputShape(
            OperationType.Flatten,
            inputShapes);

        // Assert
        Assert.Equal(2, outputShape.Length);
        Assert.Equal(32, outputShape[0]);
        Assert.Equal(3 * 224 * 224, outputShape[1]);
    }

    #endregion

    #region Pooling Tests

    [Fact]
    public void InferOutputShape_MaxPool2D_CorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 64, 112, 112 } };
        var parameters = new Dictionary<string, object>
        {
            { "kernel_size", 2 },
            { "stride", 2 },
            { "padding", 0 }
        };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.MaxPool2D,
            inputShapes,
            parameters);

        // Assert
        Assert.Equal(4, outputShape.Length);
        Assert.Equal(32, outputShape[0]);
        Assert.Equal(64, outputShape[1]);
        Assert.Equal(56, outputShape[2]); // (112 + 2*0 - 2) / 2 + 1
        Assert.Equal(56, outputShape[3]);
    }

    [Fact]
    public void InferOutputShape_AveragePool2D_CorrectShape()
    {
        // Arrange
        var inputShapes = new[] { new long[] { 32, 64, 112, 112 } };
        var parameters = new Dictionary<string, object>
        {
            { "kernel_size", 2 },
            { "stride", 2 },
            { "padding", 0 }
        };

        // Act
        var outputShape = _engine.InferOutputShape(
            OperationType.AveragePool2D,
            inputShapes,
            parameters);

        // Assert
        Assert.Equal(4, outputShape.Length);
        Assert.Equal(32, outputShape[0]);
        Assert.Equal(64, outputShape[1]);
        Assert.Equal(56, outputShape[2]);
        Assert.Equal(56, outputShape[3]);
    }

    #endregion
}
