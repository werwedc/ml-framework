using Xunit;
using RitterFramework.Core.Diagnostics;

namespace RitterFramework.Core.Tests.Diagnostics;

/// <summary>
/// Unit tests for MLFrameworkDiagnostics.
/// </summary>
public class MLFrameworkDiagnosticsTests
{
    public MLFrameworkDiagnosticsTests()
    {
        // Reset diagnostics state before each test
        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    [Fact]
    public void IsEnabled_ShouldBeFalseByDefault()
    {
        // Act & Assert
        Assert.False(MLFrameworkDiagnostics.IsEnabled);
    }

    [Fact]
    public void IsVerbose_ShouldBeFalseByDefault()
    {
        // Act & Assert
        Assert.False(MLFrameworkDiagnostics.IsVerbose);
    }

    [Fact]
    public void EnableDiagnostics_ShouldSetIsEnabledToTrue()
    {
        // Act
        MLFrameworkDiagnostics.EnableDiagnostics();

        // Assert
        Assert.True(MLFrameworkDiagnostics.IsEnabled);
    }

    [Fact]
    public void DisableDiagnostics_ShouldSetIsEnabledToFalse()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        // Act
        MLFrameworkDiagnostics.DisableDiagnostics();

        // Assert
        Assert.False(MLFrameworkDiagnostics.IsEnabled);
    }

    [Fact]
    public void EnableDiagnostics_WithVerbose_ShouldSetIsVerboseToTrue()
    {
        // Act
        MLFrameworkDiagnostics.EnableDiagnostics(verbose: true);

        // Assert
        Assert.True(MLFrameworkDiagnostics.IsVerbose);
    }

    [Fact]
    public void DisableDiagnostics_ShouldResetIsVerbose()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics(verbose: true);

        // Act
        MLFrameworkDiagnostics.DisableDiagnostics();

        // Assert
        Assert.False(MLFrameworkDiagnostics.IsVerbose);
    }

    [Fact]
    public void CheckShapes_WhenDisabled_ShouldPerformBasicValidation()
    {
        // Arrange
        var tensor1 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 2, 2 });
        var tensor2 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 2, 2 });
        var tensors = new[] { tensor1, tensor2 };

        // Act
        var result = MLFrameworkDiagnostics.CheckShapes(OperationType.MatrixMultiply, tensors);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void CheckShapes_WhenDisabled_WithInvalidTensor_ShouldReturnFalse()
    {
        // Arrange
        var tensor1 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4 }, new int[] { 2, 2 });
        var tensors = new[] { tensor1, null! };

        // Act
        var result = MLFrameworkDiagnostics.CheckShapes(OperationType.MatrixMultiply, tensors);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void CheckShapes_WhenEnabled_WithValidShapes_ShouldReturnTrue()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var tensor1 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });
        var tensor2 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 3, 2 });
        var tensors = new[] { tensor1, tensor2 };

        // Act
        var result = MLFrameworkDiagnostics.CheckShapes(OperationType.MatrixMultiply, tensors);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void CheckShapes_WhenEnabled_WithInvalidShapes_ShouldReturnFalse()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var tensor1 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });
        var tensor2 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 }); // Incompatible shapes
        var tensors = new[] { tensor1, tensor2 };

        // Act
        var result = MLFrameworkDiagnostics.CheckShapes(OperationType.MatrixMultiply, tensors);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void GetShapeDiagnostics_WithValidShapes_ShouldReturnValidInfo()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var tensor1 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });
        var tensor2 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 3, 2 });
        var tensors = new[] { tensor1, tensor2 };

        // Act
        var diagnostics = MLFrameworkDiagnostics.GetShapeDiagnostics(
            OperationType.MatrixMultiply,
            tensors,
            "test_layer");

        // Assert
        Assert.NotNull(diagnostics);
        Assert.Equal(OperationType.MatrixMultiply, diagnostics.OperationType);
        Assert.Equal("test_layer", diagnostics.LayerName);
        Assert.True(diagnostics.IsValid);
        Assert.NotNull(diagnostics.InputShapes);
        Assert.Equal(2, diagnostics.InputShapes.Length);
        Assert.NotNull(diagnostics.ActualOutputShape);
    }

    [Fact]
    public void GetShapeDiagnostics_WithInvalidShapes_ShouldReturnInvalidInfo()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var tensor1 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });
        var tensor2 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 }); // Incompatible shapes
        var tensors = new[] { tensor1, tensor2 };

        // Act
        var diagnostics = MLFrameworkDiagnostics.GetShapeDiagnostics(
            OperationType.MatrixMultiply,
            tensors,
            "test_layer");

        // Assert
        Assert.NotNull(diagnostics);
        Assert.False(diagnostics.IsValid);
        Assert.NotNull(diagnostics.Errors);
        Assert.NotEmpty(diagnostics.Errors);
    }

    [Fact]
    public void GetShapeDiagnostics_WithoutLayerName_ShouldUseUnknown()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var tensor1 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });
        var tensor2 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 3, 2 });
        var tensors = new[] { tensor1, tensor2 };

        // Act
        var diagnostics = MLFrameworkDiagnostics.GetShapeDiagnostics(
            OperationType.MatrixMultiply,
            tensors);

        // Assert
        Assert.Equal("unknown", diagnostics.LayerName);
    }

    [Fact]
    public void GetContextualShapeDiagnostics_ShouldIncludePreviousLayerInfo()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var tensor1 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });
        var tensor2 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 3, 2 });
        var tensors = new[] { tensor1, tensor2 };

        // Act
        var diagnostics = MLFrameworkDiagnostics.GetContextualShapeDiagnostics(
            OperationType.MatrixMultiply,
            tensors,
            "layer2",
            "layer1",
            new long[] { 2, 3 });

        // Assert
        Assert.Equal("layer1", diagnostics.PreviousLayerName);
        Assert.NotNull(diagnostics.PreviousLayerShape);
        Assert.Equal(2, diagnostics.PreviousLayerShape.Length);
    }

    [Fact]
    public void GetContextualShapeDiagnostics_WithoutPreviousLayer_ShouldBeNull()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var tensor1 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });
        var tensor2 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 3, 2 });
        var tensors = new[] { tensor1, tensor2 };

        // Act
        var diagnostics = MLFrameworkDiagnostics.GetContextualShapeDiagnostics(
            OperationType.MatrixMultiply,
            tensors,
            "layer2");

        // Assert
        Assert.Null(diagnostics.PreviousLayerName);
        Assert.Null(diagnostics.PreviousLayerShape);
    }

    [Fact]
    public void GenerateSuggestedFixes_WithValidDiagnostics_ShouldReturnEmptyList()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var tensor1 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });
        var tensor2 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 3, 2 });
        var tensors = new[] { tensor1, tensor2 };

        var diagnostics = MLFrameworkDiagnostics.GetShapeDiagnostics(
            OperationType.MatrixMultiply,
            tensors);

        // Act
        var fixes = MLFrameworkDiagnostics.GenerateSuggestedFixes(diagnostics);

        // Assert
        Assert.NotNull(fixes);
        Assert.Empty(fixes);
    }

    [Fact]
    public void GenerateSuggestedFixes_WithDimensionMismatch_ShouldReturnFixes()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var tensor1 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });
        var tensor2 = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 }); // Incompatible
        var tensors = new[] { tensor1, tensor2 };

        var diagnostics = MLFrameworkDiagnostics.GetShapeDiagnostics(
            OperationType.MatrixMultiply,
            tensors);

        // Act
        var fixes = MLFrameworkDiagnostics.GenerateSuggestedFixes(diagnostics);

        // Assert
        Assert.NotNull(fixes);
        Assert.NotEmpty(fixes);
    }

    [Fact]
    public void GenerateSuggestedFixes_ForMatrixMultiply_ShouldProvideSpecificFixes()
    {
        // Arrange
        var diagnostics = new ShapeDiagnosticsInfo
        {
            OperationType = OperationType.MatrixMultiply,
            IsValid = false,
            Errors = new List<string> { "Dimension mismatch error" },
            InputShapes = new[] { new long[] { 2, 3 }, new long[] { 2, 4 } }
        };

        // Act
        var fixes = MLFrameworkDiagnostics.GenerateSuggestedFixes(diagnostics);

        // Assert
        Assert.NotNull(fixes);
        Assert.Contains(fixes, f => f.Contains("Adjust weight matrix"));
    }

    [Fact]
    public void GenerateSuggestedFixes_ForConv2D_ShouldProvideSpecificFixes()
    {
        // Arrange
        var diagnostics = new ShapeDiagnosticsInfo
        {
            OperationType = OperationType.Conv2D,
            IsValid = false,
            Errors = new List<string> { "Shape mismatch error" }
        };

        // Act
        var fixes = MLFrameworkDiagnostics.GenerateSuggestedFixes(diagnostics);

        // Assert
        Assert.NotNull(fixes);
        Assert.Contains(fixes, f => f.Contains("kernel size"));
        Assert.Contains(fixes, f => f.Contains("padding"));
        Assert.Contains(fixes, f => f.Contains("stride"));
    }

    [Fact]
    public void GenerateSuggestedFixes_ForConcat_ShouldProvideSpecificFixes()
    {
        // Arrange
        var diagnostics = new ShapeDiagnosticsInfo
        {
            OperationType = OperationType.Concat,
            IsValid = false,
            Errors = new List<string> { "Shape mismatch error" }
        };

        // Act
        var fixes = MLFrameworkDiagnostics.GenerateSuggestedFixes(diagnostics);

        // Assert
        Assert.NotNull(fixes);
        Assert.Contains(fixes, f => f.Contains("concatenation axis"));
    }

    [Fact]
    public void GetFormattedReport_ShouldIncludeAllInformation()
    {
        // Arrange
        var diagnostics = new ShapeDiagnosticsInfo
        {
            OperationType = OperationType.MatrixMultiply,
            LayerName = "test_layer",
            InputShapes = new[] { new long[] { 2, 3 }, new long[] { 3, 2 } },
            ExpectedShapes = new[] { new long[] { 2, 3 } },
            ActualOutputShape = new long[] { 2, 2 },
            IsValid = true,
            Errors = null,
            Warnings = null,
            RequirementsDescription = "Matrix multiplication: [batch, m] × [m, n] → [batch, n]"
        };

        // Act
        var report = diagnostics.GetFormattedReport();

        // Assert
        Assert.NotNull(report);
        Assert.Contains("test_layer", report);
        Assert.Contains("MatrixMultiply", report);
        Assert.Contains("Valid: True", report);
        Assert.Contains("Input Shapes:", report);
        Assert.Contains("Output Shape:", report);
        Assert.Contains("Requirements:", report);
    }

    [Fact]
    public void GetFormattedReport_WithErrors_ShouldIncludeErrors()
    {
        // Arrange
        var diagnostics = new ShapeDiagnosticsInfo
        {
            OperationType = OperationType.MatrixMultiply,
            LayerName = "test_layer",
            InputShapes = new[] { new long[] { 2, 3 }, new long[] { 2, 4 } },
            ActualOutputShape = null,
            IsValid = false,
            Errors = new List<string> { "Dimension mismatch error" },
            RequirementsDescription = "Test requirements"
        };

        // Act
        var report = diagnostics.GetFormattedReport();

        // Assert
        Assert.Contains("Valid: False", report);
        Assert.Contains("Errors:", report);
        Assert.Contains("Dimension mismatch error", report);
    }

    [Fact]
    public void GetFormattedReport_WithContext_ShouldIncludePreviousLayer()
    {
        // Arrange
        var diagnostics = new ShapeDiagnosticsInfo
        {
            LayerName = "layer2",
            InputShapes = new[] { new long[] { 2, 3 } },
            ActualOutputShape = new long[] { 2, 2 },
            IsValid = true,
            PreviousLayerName = "layer1",
            PreviousLayerShape = new long[] { 2, 3 }
        };

        // Act
        var report = diagnostics.GetFormattedReport();

        // Assert
        Assert.Contains("Context:", report);
        Assert.Contains("Previous layer: layer1", report);
        Assert.Contains("Previous output: [2, 3]", report);
    }

    [Fact]
    public void GetShapeString_WithValidTensor_ShouldReturnFormattedShape()
    {
        // Arrange
        var tensor = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });

        // Act
        var shapeString = tensor.GetShapeString();

        // Assert
        Assert.Equal("[2, 3]", shapeString);
    }

    [Fact]
    public void GetShapeString_WithNullTensor_ShouldReturnNull()
    {
        // Arrange
        global::RitterFramework.Core.Tensor.Tensor? tensor = null;

        // Act
        var shapeString = tensor.GetShapeString();

        // Assert
        Assert.Equal("null", shapeString);
    }

    [Fact]
    public void GetElementCount_ShouldReturnCorrectCount()
    {
        // Arrange
        var tensor = new global::RitterFramework.Core.Tensor.Tensor(new float[] { 1, 2, 3, 4, 5, 6 }, new int[] { 2, 3 });

        // Act
        var count = tensor.GetElementCount();

        // Assert
        Assert.Equal(6, count);
    }

    [Fact]
    public void GetElementCount_WithNullTensor_ShouldReturnZero()
    {
        // Arrange
        global::RitterFramework.Core.Tensor.Tensor tensor = null!;

        // Act
        var count = tensor.GetElementCount();

        // Assert
        Assert.Equal(0, count);
    }

    [Fact]
    public void Initialize_WithCustomRegistry_ShouldUseProvidedRegistry()
    {
        // Arrange
        var customRegistry = new DefaultOperationMetadataRegistry();

        // Act
        MLFrameworkDiagnostics.Initialize(registry: customRegistry);
        MLFrameworkDiagnostics.EnableDiagnostics();

        // Assert
        Assert.True(MLFrameworkDiagnostics.IsEnabled);
    }

    [Fact]
    public void Initialize_WithCustomInferenceEngine_ShouldUseProvidedEngine()
    {
        // Arrange
        var customEngine = new DefaultShapeInferenceEngine(new DefaultOperationMetadataRegistry());

        // Act
        MLFrameworkDiagnostics.Initialize(inferenceEngine: customEngine);
        MLFrameworkDiagnostics.EnableDiagnostics();

        // Assert
        Assert.True(MLFrameworkDiagnostics.IsEnabled);
    }

    [Fact]
    public void CheckShapes_ForConv2D_WithValidShapes_ShouldReturnTrue()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var inputTensor = new global::RitterFramework.Core.Tensor.Tensor(new float[32 * 3 * 224 * 224], new int[] { 32, 3, 224, 224 });
        var weightTensor = new global::RitterFramework.Core.Tensor.Tensor(new float[64 * 3 * 3 * 3], new int[] { 64, 3, 3, 3 });
        var tensors = new[] { inputTensor, weightTensor };

        // Act
        var result = MLFrameworkDiagnostics.CheckShapes(OperationType.Conv2D, tensors);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void CheckShapes_ForLinear_WithValidShapes_ShouldReturnTrue()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var inputTensor = new global::RitterFramework.Core.Tensor.Tensor(new float[32 * 128], new int[] { 32, 128 });
        var weightTensor = new global::RitterFramework.Core.Tensor.Tensor(new float[256 * 128], new int[] { 256, 128 });
        var tensors = new[] { inputTensor, weightTensor };

        // Act
        var result = MLFrameworkDiagnostics.CheckShapes(OperationType.Linear, tensors);

        // Assert
        Assert.True(result);
    }
}
