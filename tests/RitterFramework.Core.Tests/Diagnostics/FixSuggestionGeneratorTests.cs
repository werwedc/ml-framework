using RitterFramework.Core;
using RitterFramework.Core.Diagnostics;
using Xunit;

namespace RitterFramework.Core.Tests.Diagnostics;

public class FixSuggestionGeneratorTests
{
    private readonly FixSuggestionGenerator _generator;

    public FixSuggestionGeneratorTests()
    {
        _generator = new FixSuggestionGenerator();
    }

    [Fact]
    public void GenerateSuggestions_WithNullInputShapes_ReturnsEmptyList()
    {
        // Arrange
        long[][] inputShapes = null;
        long[][] expectedShapes = Array.Empty<long[]>();

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.Linear,
            inputShapes,
            expectedShapes);

        // Assert
        Assert.Empty(result);
    }

    [Fact]
    public void GenerateSuggestions_WithEmptyInputShapes_ReturnsEmptyList()
    {
        // Arrange
        long[][] inputShapes = Array.Empty<long[]>();
        long[][] expectedShapes = Array.Empty<long[]>();

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.Linear,
            inputShapes,
            expectedShapes);

        // Assert
        Assert.Empty(result);
    }

    #region Pattern 1: Missing Batch Dimension

    [Fact]
    public void GenerateSuggestions_MissingBatchDimension_AddsCorrectSuggestion()
    {
        // Arrange
        // Input: [784], Expected: [*, 784]
        long[][] inputShapes = new[] { new long[] { 784 } };
        long[][] expectedShapes = new[] { new long[] { 32, 784 } };

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.Reshape,
            inputShapes,
            expectedShapes);

        // Assert
        Assert.Contains(result, s => s.Contains("missing batch dimension"));
        Assert.Contains(result, s => s.Contains("unsqueeze(0)") || s.Contains("reshape to [1"));
    }

    [Fact]
    public void GenerateSuggestions_MissingBatchDimension3DTo4D_AddsCorrectSuggestion()
    {
        // Arrange
        // Input: [224, 224, 3], Expected: [32, 224, 224, 3]
        long[][] inputShapes = new[] { new long[] { 224, 224, 3 } };
        long[][] expectedShapes = new[] { new long[] { 32, 224, 224, 3 } };

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.Reshape,
            inputShapes,
            expectedShapes);

        // Assert
        Assert.Contains(result, s => s.Contains("missing batch dimension"));
    }

    #endregion

    #region Pattern 2: Channel Order Mismatch

    [Fact]
    public void GenerateSuggestions_ChannelOrderMismatchNHWCtoNCHW_AddsCorrectSuggestion()
    {
        // Arrange
        // Input: [32, 224, 224, 3], Expected: [32, 3, 224, 224]
        long[][] inputShapes = new[] { new long[] { 32, 224, 224, 3 } };
        long[][] expectedShapes = new[] { new long[] { 32, 3, 224, 224 } };

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.Conv2D,
            inputShapes,
            expectedShapes);

        // Assert
        Assert.Contains(result, s => s.Contains("Channel order mismatch"));
        Assert.Contains(result, s => s.Contains("NHWC") && s.Contains("NCHW"));
    }

    [Fact]
    public void GenerateSuggestions_ChannelOrderMismatchNCHWtoNHWC_AddsCorrectSuggestion()
    {
        // Arrange
        // Input: [32, 3, 224, 224], Expected: [32, 224, 224, 3]
        long[][] inputShapes = new[] { new long[] { 32, 3, 224, 224 } };
        long[][] expectedShapes = new[] { new long[] { 32, 224, 224, 3 } };

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.Conv2D,
            inputShapes,
            expectedShapes);

        // Assert
        Assert.Contains(result, s => s.Contains("Channel order mismatch"));
    }

    [Fact]
    public void GenerateSuggestions_NonConv2DOperation_DoesNotDetectChannelOrder()
    {
        // Arrange
        long[][] inputShapes = new[] { new long[] { 32, 224, 224, 3 } };
        long[][] expectedShapes = new[] { new long[] { 32, 3, 224, 224 } };

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.Linear,
            inputShapes,
            expectedShapes);

        // Assert
        Assert.DoesNotContain(result, s => s.Contains("Channel order mismatch"));
    }

    #endregion

    #region Pattern 3: Feature Size Mismatch in Linear Layer

    [Fact]
    public void GenerateSuggestions_FeatureSizeMismatch_AddsCorrectSuggestion()
    {
        // Arrange
        // Input: [32, 256], Expected: [32, 128]
        long[][] inputShapes = new[] { new long[] { 32, 256 } };
        long[][] expectedShapes = new[] { new long[] { 32, 128 } };

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.Linear,
            inputShapes,
            expectedShapes);

        // Assert
        Assert.Contains(result, s => s.Contains("Previous layer outputs 256 features"));
        Assert.Contains(result, s => s.Contains("expects 128"));
        Assert.Contains(result, s => s.Contains("Adjust layer configuration"));
    }

    [Fact]
    public void GenerateSuggestions_FeatureSizeMatch_DoesNotAddSuggestion()
    {
        // Arrange
        long[][] inputShapes = new[] { new long[] { 32, 256 } };
        long[][] expectedShapes = new[] { new long[] { 32, 256 } };

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.Linear,
            inputShapes,
            expectedShapes);

        // Assert
        Assert.DoesNotContain(result, s => s.Contains("Previous layer outputs"));
    }

    #endregion

    #region Pattern 4: Transpose Required

    [Fact]
    public void GenerateSuggestions_TransposeRequiredLinear_AddsCorrectSuggestion()
    {
        // Arrange
        // Input: [10, 32], where it might need transpose
        long[][] inputShapes = new[] { new long[] { 10, 32 } };
        long[][] expectedShapes = new[] { new long[] { 32, 10 } };

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.Linear,
            inputShapes,
            expectedShapes);

        // Assert
        Assert.Contains(result, s => s.Contains("Consider transposing"));
    }

    [Fact]
    public void GenerateSuggestions_TransposeRequiredMatMul_AddsCorrectSuggestion()
    {
        // Arrange
        long[][] inputShapes = new[] { new long[] { 10, 32 } };
        long[][] expectedShapes = new[] { new long[] { 10, 64 } };

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.MatrixMultiply,
            inputShapes,
            expectedShapes);

        // Assert
        Assert.Contains(result, s => s.Contains("Consider transposing"));
    }

    #endregion

    #region Pattern 5: Concatenation Dimension Mismatch

    [Fact]
    public void GenerateSuggestions_ConcatDimensionMismatch_AddsCorrectSuggestion()
    {
        // Arrange
        // Input1: [32, 128], Input2: [32, 256], Axis: 0 mismatch
        long[][] inputShapes = new[]
        {
            new long[] { 32, 128 },
            new long[] { 32, 256 }
        };

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.Concat,
            inputShapes,
            null);

        // Assert
        Assert.Contains(result, s => s.Contains("Cannot concatenate on axis"));
        Assert.Contains(result, s => s.Contains("Use axis 1") || s.Contains("reshape inputs"));
    }

    [Fact]
    public void GenerateSuggestions_ConcatSingleInput_ReturnsEmptySuggestions()
    {
        // Arrange
        long[][] inputShapes = new[]
        {
            new long[] { 32, 128 }
        };

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.Concat,
            inputShapes,
            null);

        // Assert
        Assert.Empty(result);
    }

    #endregion

    #region Pattern 6: Broadcasting Failure

    [Fact]
    public void GenerateSuggestions_BroadcastingFailure_AddsCorrectSuggestion()
    {
        // Arrange
        // Input1: [32, 10], Input2: [20, 10] - cannot broadcast
        long[][] inputShapes = new[]
        {
            new long[] { 32, 10 },
            new long[] { 20, 10 }
        };

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.Broadcast,
            inputShapes,
            null);

        // Assert
        Assert.Contains(result, s => s.Contains("Cannot broadcast shapes"));
        Assert.Contains(result, s => s.Contains("[32, 10]") && s.Contains("[20, 10]"));
        Assert.Contains(result, s => s.Contains("Batch sizes must match or be 1"));
    }

    [Fact]
    public void GenerateSuggestions_BroadcastingSuccess_NoFailureSuggestion()
    {
        // Arrange
        // Input1: [32, 10], Input2: [1, 10] - can broadcast
        long[][] inputShapes = new[]
        {
            new long[] { 32, 10 },
            new long[] { 1, 10 }
        };

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.Broadcast,
            inputShapes,
            null);

        // Assert
        Assert.DoesNotContain(result, s => s.Contains("Cannot broadcast shapes"));
    }

    #endregion

    #region Edge Cases and Generic Patterns

    [Fact]
    public void GenerateSuggestions_SqueezeOpportunity_AddsCorrectSuggestion()
    {
        // Arrange
        long[][] inputShapes = new[] { new long[] { 1, 32, 256 } };

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.Reshape,
            inputShapes,
            null);

        // Assert
        Assert.Contains(result, s => s.Contains("squeeze dimension"));
    }

    [Fact]
    public void GenerateSuggestions_MultiplePatterns_ReturnsAllSuggestions()
    {
        // Arrange
        // Input: [784], Expected: [32, 784]
        long[][] inputShapes = new[] { new long[] { 784 } };
        long[][] expectedShapes = new[] { new long[] { 32, 784 } };

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.Reshape,
            inputShapes,
            expectedShapes);

        // Assert
        Assert.NotEmpty(result);
        // Should have at least the missing batch dimension suggestion
        Assert.Contains(result, s => s.Contains("missing batch dimension"));
    }

    [Fact]
    public void GenerateSuggestions_GenericOperation_AddsGenericSuggestion()
    {
        // Arrange
        long[][] inputShapes = new[] { new long[] { 10, 20, 30 } };
        long[][] expectedShapes = new[] { new long[] { 10, 30, 20 } };

        // Act
        var result = _generator.GenerateSuggestions(
            OperationType.ReduceSum,
            inputShapes,
            expectedShapes);

        // Assert
        Assert.Contains(result, s => s.Contains("Shape mismatch"));
    }

    [Fact]
    public void GenerateSuggestions_UnknownOperation_DoesNotThrow()
    {
        // Arrange
        long[][] inputShapes = new[] { new long[] { 10, 20 } };

        // Act & Assert
        var result = _generator.GenerateSuggestions(
            OperationType.Stack,
            inputShapes,
            null);

        Assert.NotNull(result);
    }

    #endregion
}
