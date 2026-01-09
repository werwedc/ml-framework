using RitterFramework.Core;
using RitterFramework.Core.Diagnostics;
using RitterFramework.Core.Tensor;
using Xunit;

namespace MLFramework.Tests.Diagnostics;

/// <summary>
/// Unit tests for MLFrameworkDiagnostics class.
/// </summary>
public class MLFrameworkDiagnosticsTests
{
    public MLFrameworkDiagnosticsTests()
    {
        // Ensure diagnostics is disabled after each test
        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    [Fact]
    public void EnableDiagnostics_SetsIsEnabledToTrue()
    {
        // Act
        MLFrameworkDiagnostics.EnableDiagnostics();

        // Assert
        Assert.True(MLFrameworkDiagnostics.IsEnabled);

        // Cleanup
        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    [Fact]
    public void DisableDiagnostics_SetsIsEnabledToFalse()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        // Act
        MLFrameworkDiagnostics.DisableDiagnostics();

        // Assert
        Assert.False(MLFrameworkDiagnostics.IsEnabled);
    }

    [Fact]
    public void IsVerbose_SetCorrectly()
    {
        // Act
        MLFrameworkDiagnostics.EnableDiagnostics(verbose: true);

        // Assert
        Assert.True(MLFrameworkDiagnostics.IsVerbose);

        // Act
        MLFrameworkDiagnostics.EnableDiagnostics(verbose: false);

        // Assert
        Assert.False(MLFrameworkDiagnostics.IsVerbose);

        // Cleanup
        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    [Fact]
    public void CheckShapes_WithValidShapes_ReturnsTrue()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();
        var tensor1 = CreateTestTensor(new[] { 32, 10 });
        var tensor2 = CreateTestTensor(new[] { 10, 5 });

        // Act
        var result = MLFrameworkDiagnostics.CheckShapes(
            OperationType.MatrixMultiply,
            new[] { tensor1, tensor2 });

        // Assert
        Assert.True(result);

        // Cleanup
        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    [Fact]
    public void CheckShapes_WithInvalidShapes_ReturnsFalse()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();
        var tensor1 = CreateTestTensor(new[] { 32, 10 });
        var tensor2 = CreateTestTensor(new[] { 5, 10 });

        // Act
        var result = MLFrameworkDiagnostics.CheckShapes(
            OperationType.MatrixMultiply,
            new[] { tensor1, tensor2 });

        // Assert
        Assert.False(result);

        // Cleanup
        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    [Fact]
    public void GetShapeDiagnostics_ReturnsCorrectInfo()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var tensor1 = CreateTestTensor(new[] { 32L, 256L });
        var tensor2 = CreateTestTensor(new[] { 128L, 10L });

        // Act
        var diagnostics = MLFrameworkDiagnostics.GetShapeDiagnostics(
            OperationType.MatrixMultiply,
            new[] { tensor1, tensor2 },
            "test_layer");

        // Assert
        Assert.NotNull(diagnostics);
        Assert.Equal(OperationType.MatrixMultiply, diagnostics.OperationType);
        Assert.Equal("test_layer", diagnostics.LayerName);
        Assert.False(diagnostics.IsValid);
        Assert.NotEmpty(diagnostics.Errors);

        // Cleanup
        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    [Fact]
    public void GetShapeDiagnostics_WithValidShapes_ReturnsValidDiagnostics()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();
        var tensor1 = CreateTestTensor(new[] { 32, 10 });
        var tensor2 = CreateTestTensor(new[] { 10, 5 });

        // Act
        var diagnostics = MLFrameworkDiagnostics.GetShapeDiagnostics(
            OperationType.MatrixMultiply,
            new[] { tensor1, tensor2 });

        // Assert
        Assert.NotNull(diagnostics);
        Assert.True(diagnostics.IsValid);
        Assert.Empty(diagnostics.Errors);

        // Cleanup
        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    [Fact]
    public void GenerateSuggestedFixes_ReturnsAppropriateSuggestions()
    {
        // Arrange
        var diagnostics = new ShapeDiagnosticsInfo
        {
            OperationType = OperationType.MatrixMultiply,
            InputShapes = new[] { new long[] { 32, 256 }, new long[] { 128, 10 } },
            Errors = new List<string> { "Dimension mismatch" },
            IsValid = false
        };

        // Act
        var fixes = MLFrameworkDiagnostics.GenerateSuggestedFixes(diagnostics);

        // Assert
        Assert.NotNull(fixes);
        Assert.NotEmpty(fixes);
        Assert.True(fixes.Any(f => f.Contains("weight") || f.Contains("matrix")));
    }

    [Fact]
    public void GenerateSuggestedFixes_WithValidDiagnostics_ReturnsEmptyList()
    {
        // Arrange
        var diagnostics = new ShapeDiagnosticsInfo
        {
            IsValid = true,
            Errors = new List<string>()
        };

        // Act
        var fixes = MLFrameworkDiagnostics.GenerateSuggestedFixes(diagnostics);

        // Assert
        Assert.NotNull(fixes);
        Assert.Empty(fixes);
    }

    [Fact]
    public void GetContextualShapeDiagnostics_IncludesPreviousLayerContext()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();
        var tensor1 = CreateTestTensor(new[] { 32, 256 });
        var tensor2 = CreateTestTensor(new[] { 128, 10 });

        // Act
        var diagnostics = MLFrameworkDiagnostics.GetContextualShapeDiagnostics(
            OperationType.MatrixMultiply,
            new[] { tensor1, tensor2 },
            "fc2",
            "fc1",
            new long[] { 32, 256 });

        // Assert
        Assert.NotNull(diagnostics);
        Assert.Equal("fc1", diagnostics.PreviousLayerName);
        Assert.Equal(new long[] { 32, 256 }, diagnostics.PreviousLayerShape);

        // Cleanup
        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    [Fact]
    public void CheckShapes_WhenDisabled_PerformsBasicValidation()
    {
        // Arrange
        MLFrameworkDiagnostics.DisableDiagnostics();
        var tensor1 = CreateTestTensor(new[] { 32, 10 });
        var tensor2 = CreateTestTensor(new[] { 10, 5 });

        // Act
        var result = MLFrameworkDiagnostics.CheckShapes(
            OperationType.MatrixMultiply,
            new[] { tensor1, tensor2 });

        // Assert - Should still return true for valid tensors
        Assert.True(result);
    }

    [Fact]
    public void Initialize_WithCustomRegistry_UseProvidedRegistry()
    {
        // Arrange
        var customRegistry = new DefaultOperationMetadataRegistry();
        customRegistry.RegisterOperation(OperationType.MatrixMultiply, new OperationShapeRequirements
        {
            InputCount = 2,
            Description = "Custom registry"
        });

        // Act
        MLFrameworkDiagnostics.Initialize(registry: customRegistry);
        MLFrameworkDiagnostics.EnableDiagnostics();

        // Assert - Should use the custom registry
        Assert.True(MLFrameworkDiagnostics.IsEnabled);

        // Cleanup
        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    [Fact]
    public void Initialize_WithCustomInferenceEngine_UseProvidedEngine()
    {
        // Arrange
        var customRegistry = new DefaultOperationMetadataRegistry();
        var customEngine = new DefaultShapeInferenceEngine(customRegistry);

        // Act
        MLFrameworkDiagnostics.Initialize(
            registry: customRegistry,
            inferenceEngine: customEngine);
        MLFrameworkDiagnostics.EnableDiagnostics();

        // Assert - Should initialize successfully
        Assert.True(MLFrameworkDiagnostics.IsEnabled);

        // Cleanup
        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    [Fact]
    public void GenerateSuggestedFixes_Conv2D_ReturnsRelevantSuggestions()
    {
        // Arrange
        var diagnostics = new ShapeDiagnosticsInfo
        {
            OperationType = OperationType.Conv2D,
            InputShapes = new[] { new long[] { 32, 3, 224, 224 }, new long[] { 64, 64, 3, 3 } },
            Errors = new List<string> { "Channel mismatch" },
            IsValid = false
        };

        // Act
        var fixes = MLFrameworkDiagnostics.GenerateSuggestedFixes(diagnostics);

        // Assert
        Assert.NotNull(fixes);
        Assert.NotEmpty(fixes);
        Assert.True(fixes.Any(f => f.Contains("kernel") || f.Contains("padding")));
    }

    [Fact]
    public void GenerateSuggestedFixes_BatchSizeError_ReturnsBatchSizeFix()
    {
        // Arrange
        var diagnostics = new ShapeDiagnosticsInfo
        {
            OperationType = OperationType.MatrixMultiply,
            Errors = new List<string> { "Batch size mismatch" },
            IsValid = false
        };

        // Act
        var fixes = MLFrameworkDiagnostics.GenerateSuggestedFixes(diagnostics);

        // Assert
        Assert.NotNull(fixes);
        Assert.NotEmpty(fixes);
        Assert.True(fixes.Any(f => f.Contains("batch") || f.Contains("Batch")));
    }

    [Fact]
    public void GetShapeDiagnostics_WithParameters_PassesParametersToRegistry()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();
        var tensor1 = CreateTestTensor(new[] { 32, 3, 224, 224 });
        var tensor2 = CreateTestTensor(new[] { 64, 3, 3, 3 });
        var parameters = new Dictionary<string, object>
        {
            { "stride", new[] { 1, 1 } },
            { "padding", new[] { 0, 0 } }
        };

        // Act
        var diagnostics = MLFrameworkDiagnostics.GetShapeDiagnostics(
            OperationType.Conv2D,
            new[] { tensor1, tensor2 },
            parameters: parameters);

        // Assert - Should handle parameters correctly
        Assert.NotNull(diagnostics);
        Assert.Equal(OperationType.Conv2D, diagnostics.OperationType);

        // Cleanup
        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    private Tensor CreateTestTensor(long[] shape)
    {
        // Create a test tensor with the given shape
        var intShape = shape.Select(s => (int)s).ToArray();
        var size = 1;
        foreach (var dim in intShape)
        {
            size *= dim;
        }
        var data = new float[size];
        return new Tensor(data, intShape);
    }
}
