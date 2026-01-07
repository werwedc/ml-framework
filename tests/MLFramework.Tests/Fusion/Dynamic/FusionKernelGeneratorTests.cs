using MLFramework.Fusion.Dynamic;
using MLFramework.Shapes;

namespace MLFramework.Tests.Fusion.Dynamic;

/// <summary>
/// Unit tests for FusionKernelGenerator
/// </summary>
public class FusionKernelGeneratorTests
{
    [Fact]
    public void GenerateFusedKernel_WithNullNode_ThrowsArgumentNullException()
    {
        // Arrange
        var generator = new FusionKernelGenerator();
        var shapes = new List<int[]> { new[] { 128, 256 } };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => generator.GenerateFusedKernel(null!, shapes));
    }

    [Fact]
    public void GenerateFusedKernel_WithNullShapes_ThrowsArgumentException()
    {
        // Arrange
        var generator = new FusionKernelGenerator();
        var node = CreateTestFusionNode();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => generator.GenerateFusedKernel(node, null!));
    }

    [Fact]
    public void GenerateFusedKernel_WithEmptyShapes_ThrowsArgumentException()
    {
        // Arrange
        var generator = new FusionKernelGenerator();
        var node = CreateTestFusionNode();
        var shapes = new List<int[]>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => generator.GenerateFusedKernel(node, shapes));
    }

    [Fact]
    public void GenerateFusedKernel_WithValidInputs_ReturnsKernel()
    {
        // Arrange
        var generator = new FusionKernelGenerator();
        var node = CreateTestFusionNode();
        var shapes = new List<int[]> { new[] { 128, 256 } };

        // Act
        var kernel = generator.GenerateFusedKernel(node, shapes);

        // Assert
        Assert.NotNull(kernel);
        Assert.NotEmpty(kernel.KernelId);
        Assert.NotEmpty(kernel.SourceCode);
        Assert.NotNull(kernel.Binary);
        Assert.False(kernel.IsGeneric);
        Assert.Single(kernel.SpecializedShapes);
    }

    [Fact]
    public void GenerateFusedKernel_CachesKernels()
    {
        // Arrange
        var generator = new FusionKernelGenerator();
        var node = CreateTestFusionNode();
        var shapes = new List<int[]> { new[] { 128, 256 } };

        // Act
        var kernel1 = generator.GenerateFusedKernel(node, shapes);
        var kernel2 = generator.GenerateFusedKernel(node, shapes);

        // Assert
        Assert.Same(kernel1, kernel2);
        Assert.Equal(1, generator.CacheCount);
    }

    [Fact]
    public void GenerateGenericKernel_WithNullNode_ThrowsArgumentNullException()
    {
        // Arrange
        var generator = new FusionKernelGenerator();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => generator.GenerateGenericKernel(null!));
    }

    [Fact]
    public void GenerateGenericKernel_WithValidNode_ReturnsGenericKernel()
    {
        // Arrange
        var generator = new FusionKernelGenerator();
        var node = CreateTestFusionNode();

        // Act
        var kernel = generator.GenerateGenericKernel(node);

        // Assert
        Assert.NotNull(kernel);
        Assert.NotEmpty(kernel.KernelId);
        Assert.NotEmpty(kernel.SourceCode);
        Assert.NotNull(kernel.Binary);
        Assert.True(kernel.IsGeneric);
        Assert.Empty(kernel.SpecializedShapes);
    }

    [Fact]
    public void GenerateGenericKernel_CachesKernels()
    {
        // Arrange
        var generator = new FusionKernelGenerator();
        var node = CreateTestFusionNode();

        // Act
        var kernel1 = generator.GenerateGenericKernel(node);
        var kernel2 = generator.GenerateGenericKernel(node);

        // Assert
        Assert.Same(kernel1, kernel2);
        Assert.Equal(1, generator.CacheCount);
    }

    [Fact]
    public void CanGenerateSpecialized_WithValidShapes_ReturnsTrue()
    {
        // Arrange
        var generator = new FusionKernelGenerator();
        var shapes = new List<int[]>
        {
            new[] { 128, 256 },
            new[] { 64, 128 }
        };

        // Act
        var result = generator.CanGenerateSpecialized(shapes);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void CanGenerateSpecialized_WithNullShapes_ReturnsFalse()
    {
        // Arrange
        var generator = new FusionKernelGenerator();

        // Act
        var result = generator.CanGenerateSpecialized(null!);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void CanGenerateSpecialized_WithEmptyShapes_ReturnsFalse()
    {
        // Arrange
        var generator = new FusionKernelGenerator();
        var shapes = new List<int[]>();

        // Act
        var result = generator.CanGenerateSpecialized(shapes);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void CanGenerateSpecialized_WithEmptyShapeArray_ReturnsFalse()
    {
        // Arrange
        var generator = new FusionKernelGenerator();
        var shapes = new List<int[]> { Array.Empty<int>() };

        // Act
        var result = generator.CanGenerateSpecialized(shapes);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ClearCache_RemovesAllCachedKernels()
    {
        // Arrange
        var generator = new FusionKernelGenerator();
        var node = CreateTestFusionNode();
        var shapes = new List<int[]> { new[] { 128, 256 } };

        generator.GenerateFusedKernel(node, shapes);
        Assert.Equal(1, generator.CacheCount);

        // Act
        generator.ClearCache();

        // Assert
        Assert.Equal(0, generator.CacheCount);
    }

    [Fact]
    public void GenerateFusedKernel_SpecializedKernelHasSignature()
    {
        // Arrange
        var generator = new FusionKernelGenerator();
        var node = CreateTestFusionNode();
        var shapes = new List<int[]> { new[] { 128, 256 } };

        // Act
        var kernel = generator.GenerateFusedKernel(node, shapes);

        // Assert
        Assert.NotEmpty(kernel.Signature);
        Assert.Contains("specialized", kernel.Signature);
        Assert.Contains("128", kernel.Signature);
        Assert.Contains("256", kernel.Signature);
    }

    [Fact]
    public void GenerateGenericKernel_GenericKernelHasSignature()
    {
        // Arrange
        var generator = new FusionKernelGenerator();
        var node = CreateTestFusionNode();

        // Act
        var kernel = generator.GenerateGenericKernel(node);

        // Assert
        Assert.NotEmpty(kernel.Signature);
        Assert.Contains("generic", kernel.Signature);
    }

    [Fact]
    public void GenerateFusedKernel_HasEstimatedExecutionTime()
    {
        // Arrange
        var generator = new FusionKernelGenerator();
        var node = CreateTestFusionNode();
        var shapes = new List<int[]> { new[] { 128, 256 } };

        // Act
        var kernel = generator.GenerateFusedKernel(node, shapes);

        // Assert
        Assert.True(kernel.EstimatedExecutionTimeNs > 0);
    }

    [Fact]
    public void GenerateGenericKernel_SlowerThanSpecialized()
    {
        // Arrange
        var generator = new FusionKernelGenerator();
        var node = CreateTestFusionNode();
        var shapes = new List<int[]> { new[] { 128, 256 } };

        // Act
        var specialized = generator.GenerateFusedKernel(node, shapes);
        var generic = generator.GenerateGenericKernel(node);

        // Assert
        Assert.True(generic.EstimatedExecutionTimeNs >= specialized.EstimatedExecutionTimeNs);
    }

    [Fact]
    public void GenerateFusedKernel_SourceCodeContainsFusionInformation()
    {
        // Arrange
        var generator = new FusionKernelGenerator();
        var node = CreateTestFusionNode();
        var shapes = new List<int[]> { new[] { 128, 256 } };

        // Act
        var kernel = generator.GenerateFusedKernel(node, shapes);

        // Assert
        Assert.Contains(node.FusionId, kernel.SourceCode);
        Assert.Contains("Add", kernel.SourceCode);
        Assert.Contains("128", kernel.SourceCode);
        Assert.Contains("256", kernel.SourceCode);
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
