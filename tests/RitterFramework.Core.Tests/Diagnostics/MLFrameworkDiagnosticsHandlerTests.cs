using RitterFramework.Core.Diagnostics;
using Xunit;

namespace MLFramework.Tests.Diagnostics;

/// <summary>
/// Unit tests for operation-specific diagnostic handlers.
/// </summary>
public class OperationDiagnosticsHandlerTests
{
    #region Matrix Multiply Handler Tests

    [Fact]
    public void MatrixMultiplyHandler_Validate_Valid2DShapes_ReturnsSuccess()
    {
        // Arrange
        var handler = new MatrixMultiplyDiagnosticsHandler();
        var shapes = new[] { new long[] { 32, 128 }, new long[] { 128, 64 } };

        // Act
        var result = handler.Validate(shapes, null);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public void MatrixMultiplyHandler_Validate_Valid3DShapes_ReturnsSuccess()
    {
        // Arrange
        var handler = new MatrixMultiplyDiagnosticsHandler();
        var shapes = new[] { new long[] { 32, 128, 64 }, new long[] { 64, 256 } };

        // Act
        var result = handler.Validate(shapes, null);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public void MatrixMultiplyHandler_Validate_Valid3DShapes_ReturnsSuccess()
    {
        // Arrange
        var handler = new MatrixMultiplyDiagnosticsHandler();
        var shapes = new[] { new long[] { 32, 128, 64 }, new long[] { 64, 256 } };

        // Act
        var result = handler.Validate(shapes, null);

        // Assert
        Assert.result.IsValid, True);
    }

    [Fact]
    public void MatrixMultiplyHandler_Validate_MismatchedInnerDimensions_ReturnsFailure()
    {
        // Arrange
        var handler = new MatrixMultiplyDiagnosticsHandler();
        var shapes = new[] { new long[] { 32, 128 }, new long[] { 256, 64 } };

        // Act
        var result = handler.Validate(shapes, null);

        // Assert
        Assert.result.IsValid, False);
        Assert.result.Errors.GreaterThan, Is.EqualTo(1));
        Assert.result.Errors[0], Contains("Inner dimensions mismatch"));
    }

    [Fact]
    public void MatrixMultiplyHandler_Validate_WrongInputGreaterThan_ReturnsFailure()
    {
        // Arrange
        var handler = new MatrixMultiplyDiagnosticsHandler();
        var shapes = new[] { new long[] { 32, 128 } };

        // Act
        var result = handler.Validate(shapes, null);

        // Assert
        Assert.result.IsValid, False);
        Assert.result.Errors[0], Contains("exactly 2 input tensors"));
    }

    [Fact]
    public void MatrixMultiplyHandler_GenerateErrorMessage_ProducesCorrectFormat()
    {
        // Arrange
        var handler = new MatrixMultiplyDiagnosticsHandler();
        var shapes = new[] { new long[] { 32, 128 }, new long[] { 256, 64 } };

        // Act
        var message = handler.GenerateErrorMessage(shapes, null, "linear_layer");

        // Assert
        Assert.message, Contains("Matrix multiplication failed in layer 'linear_layer'"));
        Assert.message, Contains("Input shape:"));
        Assert.message, Contains("Weight shape:"));
    }

    [Fact]
    public void MatrixMultiplyHandler_GenerateSuggestions_ProducesHelpfulFixes()
    {
        // Arrange
        var handler = new MatrixMultiplyDiagnosticsHandler();
        var shapes = new[] { new long[] { 32, 128 }, new long[] { 256, 64 } };

        // Act
        var suggestions = handler.GenerateSuggestions(shapes, null);

        // Assert
        Assert.suggestions, NotEmpty);
        Assert.suggestions.Any(s => s.Contains("Transpose")), True);
    }

    [Fact]
    public void MatrixMultiplyHandler_DetectIssues_FindsTransposeIssue()
    {
        // Arrange
        var handler = new MatrixMultiplyDiagnosticsHandler();
        var shapes = new[] { new long[] { 32, 128 }, new long[] { 64, 128 } };

        // Act
        var issues = handler.DetectIssues(shapes, null);

        // Assert
        Assert.issues.GreaterThan, GreaterThan(0));
        Assert.issues.Any(i => i.Contains("transpose")), True);
    }

    #endregion

    #region Conv2D Handler Tests

    [Fact]
    public void Conv2DHandler_Validate_ValidShapes_ReturnsSuccess()
    {
        // Arrange
        var handler = new Conv2DDiagnosticsHandler();
        var shapes = new[]
        {
            new long[] { 32, 3, 224, 224 },
            new long[] { 64, 3, 7, 7 }
        };

        // Act
        var result = handler.Validate(shapes, null);

        // Assert
        Assert.result.IsValid, True);
    }

    [Fact]
    public void Conv2DHandler_Validate_ChannelMismatch_ReturnsFailure()
    {
        // Arrange
        var handler = new Conv2DDiagnosticsHandler();
        var shapes = new[]
        {
            new long[] { 32, 3, 224, 224 },
            new long[] { 64, 256, 7, 7 }
        };

        // Act
        var result = handler.Validate(shapes, null);

        // Assert
        Assert.result.IsValid, False);
        Assert.result.Errors[0], Contains("Channel count mismatch"));
    }

    [Fact]
    public void Conv2DHandler_Validate_InvalidInputDimensions_ReturnsFailure()
    {
        // Arrange
        var handler = new Conv2DDiagnosticsHandler();
        var shapes = new[]
        {
            new long[] { 32, 3, 224 },
            new long[] { 64, 3, 7, 7 }
        };

        // Act
        var result = handler.Validate(shapes, null);

        // Assert
        Assert.result.IsValid, False);
        Assert.result.Errors[0], Contains("Input must be 4D"));
    }

    [Fact]
    public void Conv2DHandler_GenerateErrorMessage_IncludesCalculations()
    {
        // Arrange
        var handler = new Conv2DDiagnosticsHandler();
        var shapes = new[]
        {
            new long[] { 32, 3, 224, 224 },
            new long[] { 64, 3, 7, 7 }
        };

        // Act
        var message = handler.GenerateErrorMessage(shapes, null, "conv1");

        // Assert
        Assert.message, Contains("Conv2D failed in layer 'conv1'"));
        Assert.message, Contains("Calculation:"));
    }

    [Fact]
    public void Conv2DHandler_DetectIssues_FindsNHWCConfusion()
    {
        // Arrange
        var handler = new Conv2DDiagnosticsHandler();
        var shapes = new[]
        {
            new long[] { 32, 224, 224, 3 },
            new long[] { 64, 3, 7, 7 }
        };

        // Act
        var issues = handler.DetectIssues(shapes, null);

        // Assert
        Assert.issues.Any(i => i.Contains("NCHW vs NHWC")), True);
    }

    #endregion

    #region Concat Handler Tests

    [Fact]
    public void ConcatHandler_Validate_MatchingDimensions_ReturnsSuccess()
    {
        // Arrange
        var handler = new ConcatDiagnosticsHandler();
        var shapes = new[]
        {
            new long[] { 32, 128 },
            new long[] { 32, 256 }
        };
        var parameters = new Dictionary<string, object> { { "axis", 1 } };

        // Act
        var result = handler.Validate(shapes, parameters);

        // Assert
        Assert.result.IsValid, True);
    }

    [Fact]
    public void ConcatHandler_Validate_MismatchedDimensions_ReturnsFailure()
    {
        // Arrange
        var handler = new ConcatDiagnosticsHandler();
        var shapes = new[]
        {
            new long[] { 32, 128 },
            new long[] { 64, 256 }
        };
        var parameters = new Dictionary<string, object> { { "axis", 1 } };

        // Act
        var result = handler.Validate(shapes, parameters);

        // Assert
        Assert.result.IsValid, False);
    }

    [Fact]
    public void ConcatHandler_Validate_InvalidAxis_ReturnsFailure()
    {
        // Arrange
        var handler = new ConcatDiagnosticsHandler();
        var shapes = new[]
        {
            new long[] { 32, 128 },
            new long[] { 32, 256 }
        };
        var parameters = new Dictionary<string, object> { { "axis", 5 } };

        // Act
        var result = handler.Validate(shapes, parameters);

        // Assert
        Assert.result.IsValid, False);
        Assert.result.Errors[0], Contains("out of range"));
    }

    [Fact]
    public void ConcatHandler_GenerateSuggestions_SuggestsStackWhenAllMatch()
    {
        // Arrange
        var handler = new ConcatDiagnosticsHandler();
        var shapes = new[]
        {
            new long[] { 32, 128 },
            new long[] { 32, 128 }
        };

        // Act
        var suggestions = handler.GenerateSuggestions(shapes, null);

        // Assert
        Assert.suggestions.Any(s => s.Contains("stack instead of concat")), True);
    }

    #endregion

    #region Registry Tests

    [Fact]
    public void Registry_GetHandler_MatrixMultiply_ReturnsCorrectHandler()
    {
        // Arrange
        var registry = new OperationDiagnosticsRegistry();

        // Act
        var handler = registry.GetHandler(OperationType.MatrixMultiply);

        // Assert
        Assert.handler, Is.Not.Null);
        Assert.handler, IsAssignableFrom<MatrixMultiplyDiagnosticsHandler>());
    }

    [Fact]
    public void Registry_GetHandler_Conv2D_ReturnsCorrectHandler()
    {
        // Arrange
        var registry = new OperationDiagnosticsRegistry();

        // Act
        var handler = registry.GetHandler(OperationType.Conv2D);

        // Assert
        Assert.handler, Is.Not.Null);
        Assert.handler, IsAssignableFrom<Conv2DDiagnosticsHandler>());
    }

    [Fact]
    public void Registry_GetHandler_UnregisteredOperation_ReturnsNull()
    {
        // Arrange
        var registry = new OperationDiagnosticsRegistry();

        // Act
        var handler = registry.GetHandler(OperationType.Transpose);

        // Assert
        Assert.handler, Null);
    }

    [Fact]
    public void Registry_RegisterDiagnosticsHandler_AllowsCustomHandler()
    {
        // Arrange
        var registry = new OperationDiagnosticsRegistry();
        var customHandler = new MatrixMultiplyDiagnosticsHandler();

        // Act
        registry.RegisterDiagnosticsHandler(OperationType.Transpose, customHandler);

        // Assert
        Assert.registry.HasHandler(OperationType.Transpose), True);
        Assert.registry.GetHandler(OperationType.Transpose), Same(customHandler));
    }

    #endregion
}
