using MLFramework.Fusion.Dynamic;
using MLFramework.Shapes;
using MLFramework.Core;

namespace MLFramework.Tests.Fusion.Dynamic;

/// <summary>
/// Unit tests for FusionNode
/// </summary>
public class FusionNodeTests
{
    [Fact]
    public void Constructor_InitializesWithDefaultValues()
    {
        // Arrange & Act
        var node = new FusionNode();

        // Assert
        Assert.NotNull(node);
        Assert.NotEmpty(node.FusionId);
        Assert.Empty(node.Operations);
        Assert.Empty(node.InputShapes);
        Assert.Empty(node.OutputShapes);
    }

    [Fact]
    public void Constructor_WithCustomId_UsesProvidedId()
    {
        // Arrange
        var customId = "test_fusion_123";

        // Act
        var node = new FusionNode(customId);

        // Assert
        Assert.Equal(customId, node.FusionId);
    }

    [Fact]
    public void Constructor_WithNullId_ThrowsArgumentNullException()
    {
        // Arrange & Act & Assert
        Assert.Throws<ArgumentNullException>(() => new FusionNode(null!));
    }

    [Fact]
    public void AddOperation_AddsOperationToList()
    {
        // Arrange
        var node = new FusionNode();
        var operation = CreateTestOperation("Add");

        // Act
        node.AddOperation(operation);

        // Assert
        Assert.Single(node.Operations);
        Assert.Equal("Add", node.Operations[0].Type);
    }

    [Fact]
    public void AddOperation_WithNullOperation_ThrowsArgumentNullException()
    {
        // Arrange
        var node = new FusionNode();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => node.AddOperation(null!));
    }

    [Fact]
    public void AddInputShape_AddsShapeToList()
    {
        // Arrange
        var node = new FusionNode();
        var shape = CreateTestShape(new[] { 128, 256 });

        // Act
        node.AddInputShape(shape);

        // Assert
        Assert.Single(node.InputShapes);
        Assert.Equal(128, node.InputShapes[0].GetDimension(0).Value);
    }

    [Fact]
    public void AddOutputShape_AddsShapeToList()
    {
        // Arrange
        var node = new FusionNode();
        var shape = CreateTestShape(new[] { 128, 256 });

        // Act
        node.AddOutputShape(shape);

        // Assert
        Assert.Single(node.OutputShapes);
        Assert.Equal(128, node.OutputShapes[0].GetDimension(0).Value);
    }

    [Fact]
    public void CanFuseWith_WithCompatibleShapes_ReturnsTrue()
    {
        // Arrange
        var node = new FusionNode();
        var shape = CreateTestShape(new[] { 128, 256 });
        node.AddOutputShape(shape);

        var nextOp = CreateTestOperation("Mul");
        var intermediateShapes = new List<SymbolicShape> { shape };

        // Act
        var result = node.CanFuseWith(nextOp, intermediateShapes);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void CanFuseWith_WithIncompatibleShapes_ReturnsFalse()
    {
        // Arrange
        var node = new FusionNode();
        var shape = CreateTestShape(new[] { 128, 256 });
        node.AddOutputShape(shape);

        var nextOp = CreateTestOperation("Mul");
        var differentShape = CreateTestShape(new[] { 64, 256 });
        var intermediateShapes = new List<SymbolicShape> { differentShape };

        // Act
        var result = node.CanFuseWith(nextOp, intermediateShapes);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void CanFuseWith_WithNullNextOp_ReturnsFalse()
    {
        // Arrange
        var node = new FusionNode();
        var shape = CreateTestShape(new[] { 128, 256 });
        node.AddOutputShape(shape);
        var intermediateShapes = new List<SymbolicShape> { shape };

        // Act
        var result = node.CanFuseWith(null!, intermediateShapes);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void GetFusedSignature_ReturnsCorrectSignature()
    {
        // Arrange
        var node = new FusionNode();
        node.AddOperation(CreateTestOperation("Add"));
        node.AddOperation(CreateTestOperation("ReLU"));
        node.AddInputShape(CreateTestShape(new[] { 128, 256 }));
        node.AddOutputShape(CreateTestShape(new[] { 128, 256 }));

        // Act
        var signature = node.GetFusedSignature();

        // Assert
        Assert.Contains("Add->ReLU", signature);
        Assert.Contains("128", signature);
        Assert.Contains("256", signature);
    }

    [Fact]
    public void ValidateFusion_WithValidNode_ReturnsTrue()
    {
        // Arrange
        var node = new FusionNode();
        node.AddOperation(CreateTestOperation("Add"));
        node.AddInputShape(CreateTestShape(new[] { 128, 256 }));
        node.AddOutputShape(CreateTestShape(new[] { 128, 256 }));

        // Act
        var result = node.ValidateFusion();

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void ValidateFusion_WithNoOperations_ReturnsFalse()
    {
        // Arrange
        var node = new FusionNode();

        // Act
        var result = node.ValidateFusion();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ValidateFusion_WithNoInputShapes_ReturnsFalse()
    {
        // Arrange
        var node = new FusionNode();
        node.AddOperation(CreateTestOperation("Add"));
        node.AddOutputShape(CreateTestShape(new[] { 128, 256 }));

        // Act
        var result = node.ValidateFusion();

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void Clone_CreatesIndependentCopy()
    {
        // Arrange
        var node = new FusionNode();
        node.AddOperation(CreateTestOperation("Add"));
        node.AddInputShape(CreateTestShape(new[] { 128, 256 }));
        node.AddOutputShape(CreateTestShape(new[] { 128, 256 }));

        // Act
        var cloned = node.Clone();

        // Assert
        Assert.NotSame(node, cloned);
        Assert.NotEqual(node.FusionId, cloned.FusionId);
        Assert.Equal(node.Operations.Count, cloned.Operations.Count);
        Assert.Equal(node.InputShapes.Count, cloned.InputShapes.Count);
        Assert.Equal(node.OutputShapes.Count, cloned.OutputShapes.Count);
    }

    private Operation CreateTestOperation(string type)
    {
        return new Operation
        {
            Id = $"{type}_{Guid.NewGuid():N}",
            Type = type,
            Name = type,
            DataType = DataType.Float32,
            Layout = TensorLayout.NCHW,
            InputShape = new TensorShape { Dimensions = new[] { 128, 256 } },
            OutputShape = new TensorShape { Dimensions = new[] { 128, 256 } },
            Inputs = new[] { "input" },
            Outputs = new[] { "output" },
            Attributes = new Dictionary<string, object>()
        };
    }

    private SymbolicShape CreateTestShape(int[] dimensions)
    {
        var dims = dimensions.Select(d => SymbolicDimensionFactory.CreateKnown(d)).ToArray();
        return new SymbolicShape(dims);
    }
}
