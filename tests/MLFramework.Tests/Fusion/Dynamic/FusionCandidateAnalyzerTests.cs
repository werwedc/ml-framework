using MLFramework.Fusion.Dynamic;
using MLFramework.Shapes;
using MLFramework.Core;

namespace MLFramework.Tests.Fusion.Dynamic;

/// <summary>
/// Unit tests for FusionCandidateAnalyzer
/// </summary>
public class FusionCandidateAnalyzerTests
{
    [Fact]
    public void FindFusibleOperations_WithEmptyList_ReturnsEmptyList()
    {
        // Arrange
        var analyzer = new FusionCandidateAnalyzer();
        var operations = new List<Operation>();

        // Act
        var result = analyzer.FindFusibleOperations(operations);

        // Assert
        Assert.Empty(result);
    }

    [Fact]
    public void FindFusibleOperations_WithNullList_ReturnsEmptyList()
    {
        // Arrange
        var analyzer = new FusionCandidateAnalyzer();

        // Act
        var result = analyzer.FindFusibleOperations(null!);

        // Assert
        Assert.Empty(result);
    }

    [Fact]
    public void FindFusibleOperations_WithSingleOperation_ReturnsEmptyList()
    {
        // Arrange
        var analyzer = new FusionCandidateAnalyzer();
        var operations = new List<Operation> { CreateTestOperation("Add") };

        // Act
        var result = analyzer.FindFusibleOperations(operations);

        // Assert
        Assert.Empty(result); // Single operation cannot be fused
    }

    [Fact]
    public void FindFusibleOperations_WithMultipleOperations_ReturnsFusionNodes()
    {
        // Arrange
        var analyzer = new FusionCandidateAnalyzer();
        var operations = new List<Operation>
        {
            CreateTestOperation("Add"),
            CreateTestOperation("ReLU"),
            CreateTestOperation("Mul")
        };

        // Act
        var result = analyzer.FindFusibleOperations(operations);

        // Assert
        Assert.NotEmpty(result);
        Assert.All(result, node => Assert.True(node.ValidateFusion()));
    }

    [Fact]
    public void AnalyzeBenefit_WithNullNode_ReturnsZeroBenefit()
    {
        // Arrange
        var analyzer = new FusionCandidateAnalyzer();

        // Act
        var benefit = analyzer.AnalyzeBenefit(null!);

        // Assert
        Assert.Equal(1.0, benefit.EstimatedSpeedup);
        Assert.Equal(0, benefit.MemorySaved);
        Assert.Equal(0, benefit.KernelCountReduction);
    }

    [Fact]
    public void AnalyzeBenefit_WithSingleOperation_ReturnsZeroBenefit()
    {
        // Arrange
        var analyzer = new FusionCandidateAnalyzer();
        var node = new FusionNode();
        node.AddOperation(CreateTestOperation("Add"));

        // Act
        var benefit = analyzer.AnalyzeBenefit(node);

        // Assert
        Assert.Equal(1.0, benefit.EstimatedSpeedup);
        Assert.Equal(0, benefit.MemorySaved);
        Assert.Equal(0, benefit.KernelCountReduction);
    }

    [Fact]
    public void AnalyzeBenefit_WithMultipleOperations_ReturnsPositiveBenefit()
    {
        // Arrange
        var analyzer = new FusionCandidateAnalyzer();
        var node = new FusionNode();
        node.AddOperation(CreateTestOperation("Add", new[] { 128, 256 }, new[] { 128, 256 }));
        node.AddOperation(CreateTestOperation("ReLU", new[] { 128, 256 }, new[] { 128, 256 }));

        // Act
        var benefit = analyzer.AnalyzeBenefit(node);

        // Assert
        Assert.True(benefit.EstimatedSpeedup > 1.0);
        Assert.True(benefit.MemorySaved > 0);
        Assert.True(benefit.KernelCountReduction > 0);
    }

    [Fact]
    public void AnalyzeBenefit_WithMoreOperations_HigherBenefit()
    {
        // Arrange
        var analyzer = new FusionCandidateAnalyzer();
        var node2Ops = new FusionNode();
        node2Ops.AddOperation(CreateTestOperation("Add", new[] { 128, 256 }, new[] { 128, 256 }));
        node2Ops.AddOperation(CreateTestOperation("ReLU", new[] { 128, 256 }, new[] { 128, 256 }));

        var node3Ops = new FusionNode();
        node3Ops.AddOperation(CreateTestOperation("Add", new[] { 128, 256 }, new[] { 128, 256 }));
        node3Ops.AddOperation(CreateTestOperation("ReLU", new[] { 128, 256 }, new[] { 128, 256 }));
        node3Ops.AddOperation(CreateTestOperation("Mul", new[] { 128, 256 }, new[] { 128, 256 }));

        // Act
        var benefit2 = analyzer.AnalyzeBenefit(node2Ops);
        var benefit3 = analyzer.AnalyzeBenefit(node3Ops);

        // Assert
        Assert.True(benefit3.EstimatedSpeedup > benefit2.EstimatedSpeedup);
        Assert.True(benefit3.MemorySaved > benefit2.MemorySaved);
    }

    [Fact]
    public void IsShapePreserving_WithElementWiseOperation_ReturnsTrue()
    {
        // Arrange
        var analyzer = new FusionCandidateAnalyzer();
        var operation = CreateTestOperation("Add", new[] { 128, 256 }, new[] { 128, 256 });

        // Act
        var result = analyzer.IsShapePreserving(operation);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void IsShapePreserving_WithNonPreservingOperation_ReturnsFalse()
    {
        // Arrange
        var analyzer = new FusionCandidateAnalyzer();
        var operation = CreateTestOperation("Conv2D", new[] { 1, 3, 32, 32 }, new[] { 1, 64, 30, 30 });

        // Act
        var result = analyzer.IsShapePreserving(operation);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void IsShapePreserving_WithNullOperation_ReturnsFalse()
    {
        // Arrange
        var analyzer = new FusionCandidateAnalyzer();

        // Act
        var result = analyzer.IsShapePreserving(null!);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void RequiresRuntimeShapeCheck_WithSymbolicShapes_ReturnsTrue()
    {
        // Arrange
        var analyzer = new FusionCandidateAnalyzer();
        var node = new FusionNode();
        node.AddOperation(CreateTestOperation("Add"));
        var symbolicDim = SymbolicDimensionFactory.CreateSymbolic("batch_size");
        node.AddInputShape(new SymbolicShape(symbolicDim, SymbolicDimensionFactory.CreateKnown(256)));

        // Act
        var result = analyzer.RequiresRuntimeShapeCheck(node);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void RequiresRuntimeShapeCheck_WithConcreteShapes_ReturnsFalse()
    {
        // Arrange
        var analyzer = new FusionCandidateAnalyzer();
        var node = new FusionNode();
        node.AddOperation(CreateTestOperation("Add"));
        node.AddInputShape(new SymbolicShape(
            SymbolicDimensionFactory.CreateKnown(128),
            SymbolicDimensionFactory.CreateKnown(256)
        ));

        // Act
        var result = analyzer.RequiresRuntimeShapeCheck(node);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void RequiresRuntimeShapeCheck_WithNullNode_ReturnsFalse()
    {
        // Arrange
        var analyzer = new FusionCandidateAnalyzer();

        // Act
        var result = analyzer.RequiresRuntimeShapeCheck(null!);

        // Assert
        Assert.False(result);
    }

    private Operation CreateTestOperation(string type, int[]? inputDims = null, int[]? outputDims = null)
    {
        inputDims ??= new[] { 128, 256 };
        outputDims ??= new[] { 128, 256 };

        return new Operation
        {
            Id = $"{type}_{Guid.NewGuid():N}",
            Type = type,
            Name = type,
            DataType = DataType.Float32,
            Layout = TensorLayout.NCHW,
            InputShape = new TensorShape { Dimensions = inputDims },
            OutputShape = new TensorShape { Dimensions = outputDims },
            Inputs = new[] { "input" },
            Outputs = new[] { "output" },
            Attributes = new Dictionary<string, object>()
        };
    }
}
