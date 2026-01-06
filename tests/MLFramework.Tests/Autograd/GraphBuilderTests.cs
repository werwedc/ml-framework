using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.Autograd;
using System;

namespace MLFramework.Tests.Autograd;

public class GraphBuilderTests : IDisposable
{
    private readonly GraphBuilder _builder;

    public GraphBuilderTests()
    {
        // Create a new graph builder for each test
        _builder = new GraphBuilder();
    }

    public void Dispose()
    {
        // Clean up after each test
        _builder.Dispose();
    }

    #region Graph Construction Tests

    [Fact]
    public void CreateNode_WithValidParameters_CreatesNodeSuccessfully()
    {
        // Arrange
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var shape = new int[] { 2, 2 };
        var tensor = new Tensor(data, shape, requiresGrad: true);
        var operation = new OperationContext("TestOperation", g => new Tensor[] { g });

        // Act
        var node = _builder.CreateNode(tensor, operation);

        // Assert
        Assert.NotNull(node);
        Assert.Equal(tensor, node.OutputTensor);
        Assert.Equal(operation, node.Operation);
        Assert.Equal(0, node.Children.Count);
        Assert.True(node.IsLeaf);
        Assert.Equal(1, _builder.NodeCount);
    }

    [Fact]
    public void CreateNode_WithChildren_CreatesNodeWithDependencies()
    {
        // Arrange
        var data1 = new float[] { 1.0f, 2.0f };
        var data2 = new float[] { 3.0f, 4.0f };
        var tensor1 = new Tensor(data1, new int[] { 2 }, requiresGrad: true);
        var tensor2 = new Tensor(data2, new int[] { 2 }, requiresGrad: true);
        var operation1 = new OperationContext("Add", g => new Tensor[] { g });
        var operation2 = new OperationContext("Mul", g => new Tensor[] { g });

        // Act
        var node1 = _builder.CreateNode(tensor1, operation1);
        var node2 = _builder.CreateNode(tensor2, operation2);
        var outputTensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var operation3 = new OperationContext("Final", g => new Tensor[] { g });
        var node3 = _builder.CreateNode(outputTensor, operation3, node1, node2);

        // Assert
        Assert.Equal(2, node3.Children.Count);
        Assert.Contains(node1, node3.Children);
        Assert.Contains(node2, node3.Children);
        Assert.False(node3.IsLeaf);
        Assert.True(node1.IsLeaf);
        Assert.True(node2.IsLeaf);
        Assert.Equal(3, _builder.NodeCount);
    }

    [Fact]
    public void CreateNode_WhenDisabled_ThrowsInvalidOperationException()
    {
        // Arrange
        _builder.IsEnabled = false;
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var operation = new OperationContext("Test", g => new Tensor[] { g });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => _builder.CreateNode(tensor, operation));
    }

    #endregion

    #region Scope Management Tests

    [Fact]
    public void PushScope_PushesNodeOntoStack()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var operation = new OperationContext("Test", g => new Tensor[] { g });
        var node = _builder.CreateNode(tensor, operation);

        // Act
        _builder.PushScope(node);

        // Assert
        Assert.Equal(node, _builder.CurrentNode);
        Assert.Equal(1, _builder.NodeStack.Count);
    }

    [Fact]
    public void PopScope_RemovesNodeFromStack()
    {
        // Arrange
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var operation = new OperationContext("Test", g => new Tensor[] { g });
        var node = _builder.CreateNode(tensor, operation);
        _builder.PushScope(node);

        // Act
        var poppedNode = _builder.PopScope();

        // Assert
        Assert.Equal(node, poppedNode);
        Assert.Null(_builder.CurrentNode);
        Assert.Equal(0, _builder.NodeStack.Count);
    }

    [Fact]
    public void PushAndPopScope_MultipleNodes_MaintainsCorrectOrder()
    {
        // Arrange
        var operation = new OperationContext("Test", g => new Tensor[] { g });
        var node1 = _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
        var node2 = _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
        var node3 = _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);

        // Act
        _builder.PushScope(node1);
        _builder.PushScope(node2);
        _builder.PushScope(node3);

        // Assert - node3 should be on top
        Assert.Equal(node3, _builder.CurrentNode);

        // Act - pop node3
        _builder.PopScope();
        Assert.Equal(node2, _builder.CurrentNode);

        // Act - pop node2
        _builder.PopScope();
        Assert.Equal(node1, _builder.CurrentNode);

        // Act - pop node1
        _builder.PopScope();
        Assert.Null(_builder.CurrentNode);
    }

    [Fact]
    public void PopScope_WhenEmpty_ThrowsInvalidOperationException()
    {
        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => _builder.PopScope());
    }

    #endregion

    #region Graph Clearing Tests

    [Fact]
    public void ClearGraph_DisposesAllNodes()
    {
        // Arrange
        var operation = new OperationContext("Test", g => new Tensor[] { g });
        _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
        _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
        _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
        Assert.Equal(3, _builder.NodeCount);

        // Act
        _builder.ClearGraph();

        // Assert
        Assert.Equal(0, _builder.NodeCount);
    }

    [Fact]
    public void ClearGraph_ClearsNodeStack()
    {
        // Arrange
        var operation = new OperationContext("Test", g => new Tensor[] { g });
        var node = _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
        _builder.PushScope(node);

        // Act
        _builder.ClearGraph();

        // Assert
        Assert.Null(_builder.CurrentNode);
        Assert.Equal(0, _builder.NodeStack.Count);
    }

    #endregion

    #region Root and Leaf Node Tests

    [Fact]
    public void GetRootNodes_ReturnsCorrectNodes()
    {
        // Arrange
        var operation = new OperationContext("Test", g => new Tensor[] { g });
        var node1 = _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
        var node2 = _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
        var node3 = _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation, node1, node2);
        var node4 = _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation, node3);

        // Act
        var rootNodes = _builder.GetRootNodes();

        // Assert
        Assert.Single(rootNodes);
        Assert.Contains(node4, rootNodes);
    }

    [Fact]
    public void GetLeafNodes_ReturnsCorrectNodes()
    {
        // Arrange
        var operation = new OperationContext("Test", g => new Tensor[] { g });
        var node1 = _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
        var node2 = _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
        var node3 = _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation, node1, node2);

        // Act
        var leafNodes = _builder.GetLeafNodes();

        // Assert
        Assert.Equal(2, leafNodes.Count);
        Assert.Contains(node1, leafNodes);
        Assert.Contains(node2, leafNodes);
    }

    #endregion

    #region OperationContext Tests

    [Fact]
    public void OperationContext_SaveAndRetrieveTensor_WorksCorrectly()
    {
        // Arrange
        var context = new OperationContext("TestOperation", g => new Tensor[] { g });
        var tensor = new Tensor(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });

        // Act
        context.SaveTensor("input", tensor);
        var retrieved = context.GetSavedTensor<Tensor>("input");

        // Assert
        Assert.NotNull(retrieved);
        Assert.Equal(tensor.Shape, retrieved.Shape);
    }

    [Fact]
    public void OperationContext_SaveTensor_WithSameKey_Overwrites()
    {
        // Arrange
        var context = new OperationContext("TestOperation", g => new Tensor[] { g });
        var tensor1 = new Tensor(new float[] { 1.0f }, new int[] { 1 });
        var tensor2 = new Tensor(new float[] { 2.0f }, new int[] { 1 });

        // Act
        context.SaveTensor("key", tensor1);
        context.SaveTensor("key", tensor2);
        var retrieved = context.GetSavedTensor<Tensor>("key");

        // Assert
        Assert.Equal(tensor2, retrieved);
    }

    [Fact]
    public void OperationContext_GetSavedTensor_WithNonExistentKey_ThrowsKeyNotFoundException()
    {
        // Arrange
        var context = new OperationContext("TestOperation", g => new Tensor[] { g });

        // Act & Assert
        Assert.Throws<KeyNotFoundException>(() => context.GetSavedTensor<Tensor>("nonexistent"));
    }

    [Fact]
    public void OperationContext_GetSavedTensor_WithWrongType_ThrowsInvalidCastException()
    {
        // Arrange
        var context = new OperationContext("TestOperation", g => new Tensor[] { g });
        context.SaveTensor("scalar", 42);

        // Act & Assert
        Assert.Throws<InvalidCastException>(() => context.GetSavedTensor<Tensor>("scalar"));
    }

    [Fact]
    public void OperationContext_HasSavedTensor_ReturnsCorrectResult()
    {
        // Arrange
        var context = new OperationContext("TestOperation", g => new Tensor[] { g });
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        // Act & Assert
        Assert.False(context.HasSavedTensor("key"));

        context.SaveTensor("key", tensor);
        Assert.True(context.HasSavedTensor("key"));
    }

    [Fact]
    public void OperationContext_ClearSavedTensors_RemovesAll()
    {
        // Arrange
        var context = new OperationContext("TestOperation", g => new Tensor[] { g });
        context.SaveTensor("key1", new Tensor(new float[] { 1.0f }, new int[] { 1 }));
        context.SaveTensor("key2", new Tensor(new float[] { 2.0f }, new int[] { 1 }));

        // Act
        context.ClearSavedTensors();

        // Assert
        Assert.Equal(0, context.SavedTensorCount);
        Assert.False(context.HasSavedTensor("key1"));
        Assert.False(context.HasSavedTensor("key2"));
    }

    [Fact]
    public void OperationContext_SaveTensor_WithNull_ThrowsArgumentNullException()
    {
        // Arrange
        var context = new OperationContext("TestOperation", g => new Tensor[] { g });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => context.SaveTensor("key", null!));
    }

    [Fact]
    public void OperationContext_SaveTensor_WithEmptyKey_ThrowsArgumentException()
    {
        // Arrange
        var context = new OperationContext("TestOperation", g => new Tensor[] { g });
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => context.SaveTensor("", tensor));
    }

    [Fact]
    public void OperationContext_SavedTensorCount_ReturnsCorrectCount()
    {
        // Arrange
        var context = new OperationContext("TestOperation", g => new Tensor[] { g });

        // Act & Assert
        Assert.Equal(0, context.SavedTensorCount);

        context.SaveTensor("key1", new Tensor(new float[] { 1.0f }, new int[] { 1 }));
        Assert.Equal(1, context.SavedTensorCount);

        context.SaveTensor("key2", new Tensor(new float[] { 2.0f }, new int[] { 1 }));
        Assert.Equal(2, context.SavedTensorCount);
    }

    #endregion

    #region Thread Safety Tests

    [Fact]
    public void MultiThreadedGraphBuilding_IndependentInstances()
    {
        // Arrange
        var builder1 = new GraphBuilder();
        var builder2 = new GraphBuilder();
        var operation = new OperationContext("Test", g => new Tensor[] { g });

        // Act - Create nodes in parallel
        var task1 = Task.Run(() =>
        {
            builder1.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
            return builder1.NodeCount;
        });

        var task2 = Task.Run(() =>
        {
            builder2.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
            builder2.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
            return builder2.NodeCount;
        });

        Task.WaitAll(task1, task2);

        // Assert
        Assert.Equal(1, task1.Result);
        Assert.Equal(2, task2.Result);

        // Cleanup
        builder1.Dispose();
        builder2.Dispose();
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void BuildSimpleOperationChain_VerifiesGraphStructure()
    {
        // Arrange - Build a simple computation graph: x -> y -> z
        var operation = new OperationContext("Linear", g => new Tensor[] { g });
        var x = new Tensor(new float[] { 1.0f, 2.0f }, new int[] { 2 }, requiresGrad: true);
        var node1 = _builder.CreateNode(x, operation);

        var y = new Tensor(new float[] { 2.0f, 3.0f }, new int[] { 2 }, requiresGrad: true);
        var node2 = _builder.CreateNode(y, operation);

        var z = new Tensor(new float[] { 3.0f, 4.0f }, new int[] { 2 }, requiresGrad: true);
        var node3 = _builder.CreateNode(z, operation, node1, node2);

        // Assert - Verify graph structure
        Assert.Equal(3, _builder.NodeCount);
        Assert.Equal(2, node3.Children.Count);
        Assert.True(node1.IsLeaf);
        Assert.True(node2.IsLeaf);
        Assert.False(node3.IsLeaf);

        var roots = _builder.GetRootNodes();
        Assert.Single(roots);
        Assert.Contains(node3, roots);

        var leaves = _builder.GetLeafNodes();
        Assert.Equal(2, leaves.Count);
        Assert.Contains(node1, leaves);
        Assert.Contains(node2, leaves);
    }

    [Fact]
    public void GraphWithRequiresGradFalse_VerifiesNoNodesCreated()
    {
        // Arrange
        _builder.IsEnabled = false;
        var tensor = new Tensor(new float[] { 1.0f }, new int[] { 1 }, requiresGrad: false);
        var operation = new OperationContext("Test", g => new Tensor[] { g });

        // Act
        try
        {
            _builder.CreateNode(tensor, operation);
            Assert.True(false, "Should have thrown InvalidOperationException");
        }
        catch (InvalidOperationException)
        {
            // Expected - graph is disabled
        }

        // Assert - Verify no nodes were created
        Assert.Equal(0, _builder.NodeCount);
    }

    [Fact]
    public void ClearGraphAndRebuild_VerifiesCorrectBehavior()
    {
        // Arrange
        var operation = new OperationContext("Test", g => new Tensor[] { g });
        _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
        _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
        Assert.Equal(2, _builder.NodeCount);

        // Act - Clear graph
        _builder.ClearGraph();
        Assert.Equal(0, _builder.NodeCount);

        // Rebuild
        _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
        _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
        _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);

        // Assert
        Assert.Equal(3, _builder.NodeCount);
    }

    #endregion

    #region Disposal Tests

    [Fact]
    public void Dispose_CleansUpAllResources()
    {
        // Arrange
        var operation = new OperationContext("Test", g => new Tensor[] { g });
        _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation);
        _builder.PushScope(_builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation));

        // Act
        _builder.Dispose();

        // Assert - Operations should throw ObjectDisposedException
        Assert.Throws<ObjectDisposedException>(() => _builder.CreateNode(new Tensor(new float[] { 1.0f }, new int[] { 1 }), operation));
        Assert.Throws<ObjectDisposedException>(() => _builder.ClearGraph());
    }

    [Fact]
    public void GetCurrent_ReturnsThreadLocalBuilder()
    {
        // Arrange
        var builder = new GraphBuilder();
        var operation = new OperationContext("Test", g => new Tensor[] { g });

        // Act
        var current = GraphBuilder.GetCurrent();

        // Assert
        Assert.NotNull(current);
        Assert.Equal(builder, current);

        // Cleanup
        builder.Dispose();
    }

    #endregion
}
