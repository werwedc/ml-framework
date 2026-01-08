using MLFramework.Fusion.Dynamic;
using MLFramework.Shapes;

namespace MLFramework.Tests.Fusion.Dynamic;

/// <summary>
/// Unit tests for RuntimeShapeInjector
/// </summary>
public class RuntimeShapeInjectorTests
{
    [Fact]
    public void InjectShapeCheck_WithNullNode_ReturnsEmptyList()
    {
        // Arrange
        var injector = new RuntimeShapeInjector();

        // Act
        var result = injector.InjectShapeCheck(null!);

        // Assert
        Assert.Empty(result);
    }

    [Fact]
    public void InjectShapeCheck_WithValidNode_AddsShapeCheckOperation()
    {
        // Arrange
        var injector = new RuntimeShapeInjector();
        var node = CreateTestFusionNode();

        // Act
        var result = injector.InjectShapeCheck(node);

        // Assert
        Assert.Equal(node.Operations.Count + 1, result.Count);
        Assert.Equal("ShapeValidation", result[0].Type);
    }

    [Fact]
    public void InjectShapeCheck_PreservesOriginalOperations()
    {
        // Arrange
        var injector = new RuntimeShapeInjector();
        var node = CreateTestFusionNode();

        // Act
        var result = injector.InjectShapeCheck(node);

        // Assert
        // First is shape check, rest should be original ops
        for (int i = 0; i < node.Operations.Count; i++)
        {
            Assert.Equal(node.Operations[i].Id, result[i + 1].Id);
        }
    }

    [Fact]
    public void GenerateShapeDispatch_WithValidNode_ReturnsDispatchOperation()
    {
        // Arrange
        var injector = new RuntimeShapeInjector();
        var node = CreateTestFusionNode();

        // Act
        var result = injector.GenerateShapeDispatch(node);

        // Assert
        Assert.Equal("ShapeDispatch", result.Type);
        Assert.Contains(node.FusionId, result.Id);
        Assert.True(result.Attributes.ContainsKey("FusionId"));
        Assert.True(result.Attributes.ContainsKey("Signature"));
    }

    [Fact]
    public void GenerateShapeDispatch_WithNullNode_ThrowsArgumentNullException()
    {
        // Arrange
        var injector = new RuntimeShapeInjector();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => injector.GenerateShapeDispatch(null!));
    }

    [Fact]
    public void GenerateGenericFallback_WithValidNode_ReturnsFallbackOperation()
    {
        // Arrange
        var injector = new RuntimeShapeInjector();
        var node = CreateTestFusionNode();

        // Act
        var result = injector.GenerateGenericFallback(node);

        // Assert
        Assert.Equal("GenericFusedKernel", result.Type);
        Assert.Contains(node.FusionId, result.Id);
        Assert.True(result.Attributes.ContainsKey("FusionId"));
        Assert.True(result.Attributes.ContainsKey("OperationCount"));
        Assert.True(result.Attributes.ContainsKey("KernelType"));
    }

    [Fact]
    public void GenerateGenericFallback_WithNullNode_ThrowsArgumentNullException()
    {
        // Arrange
        var injector = new RuntimeShapeInjector();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => injector.GenerateGenericFallback(null!));
    }

    [Fact]
    public void GenerateGenericFallback_WithMultipleOperations_CorrectOperationCount()
    {
        // Arrange
        var injector = new RuntimeShapeInjector();
        var node = new FusionNode();
        node.AddOperation(CreateTestOperation("Add"));
        node.AddOperation(CreateTestOperation("ReLU"));
        node.AddOperation(CreateTestOperation("Mul"));

        // Act
        var result = injector.GenerateGenericFallback(node);

        // Assert
        Assert.Equal(3, result.Attributes["OperationCount"]);
    }

    [Fact]
    public void GenerateShapeDispatch_SignatureMatchesFusionSignature()
    {
        // Arrange
        var injector = new RuntimeShapeInjector();
        var node = CreateTestFusionNode();
        var expectedSignature = node.GetFusedSignature();

        // Act
        var result = injector.GenerateShapeDispatch(node);

        // Assert
        Assert.Equal(expectedSignature, result.Attributes["Signature"]);
    }

    private FusionNode CreateTestFusionNode()
    {
        var node = new FusionNode();
        node.AddOperation(CreateTestOperation("Add"));
        node.AddOperation(CreateTestOperation("ReLU"));
        node.AddInputShape(new SymbolicShape(
            SymbolicDimensionFactory.CreateKnown(128),
            SymbolicDimensionFactory.CreateKnown(256)
        ));
        node.AddOutputShape(new SymbolicShape(
            SymbolicDimensionFactory.CreateKnown(128),
            SymbolicDimensionFactory.CreateKnown(256)
        ));
        return node;
    }

    private MLFramework.Core.Operation CreateTestOperation(string type)
    {
        return new MLFramework.Core.Operation
        {
            Id = $"{type}_{Guid.NewGuid():N}",
            Type = type,
            Name = type,
            DataType = MLFramework.Core.DataType.Float32,
            Layout = MLFramework.Core.TensorLayout.NCHW,
            InputShape = new MLFramework.Fusion.TensorShape { Dimensions = new[] { 128, 256 } },
            OutputShape = new MLFramework.Fusion.TensorShape { Dimensions = new[] { 128, 256 } },
            Inputs = new[] { "input" },
            Outputs = new[] { "output" },
            Attributes = new Dictionary<string, object>()
        };
    }
}
