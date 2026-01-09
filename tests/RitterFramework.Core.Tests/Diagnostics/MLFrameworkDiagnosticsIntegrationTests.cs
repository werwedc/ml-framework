using RitterFramework.Core.Diagnostics;

namespace MLFramework.Tests.Diagnostics;

/// <summary>
/// Integration tests for operation diagnostics system.
/// </summary>
[TestFixture]
public class OperationDiagnosticsIntegrationTests
{
    private IOperationDiagnosticsRegistry _registry;

    [SetUp]
    public void Setup()
    {
        _registry = new OperationDiagnosticsRegistry();
    }

    #region Registry Routing Tests

    [Fact]
    public void Registry_RoutesToCorrectHandler_MatrixMultiply()
    {
        // Arrange
        var shapes = new[] { new long[] { 32, 128 }, new long[] { 128, 64 } };

        // Act
        var handler = _registry.GetHandler(OperationType.MatrixMultiply);
        var result = handler.Validate(shapes, null);

        // Assert
        Assert.result.IsValid, True);
    }

    [Fact]
    public void Registry_RoutesToCorrectHandler_Conv2D()
    {
        // Arrange
        var shapes = new[]
        {
            new long[] { 32, 3, 224, 224 },
            new long[] { 64, 3, 7, 7 }
        };

        // Act
        var handler = _registry.GetHandler(OperationType.Conv2D);
        var result = handler.Validate(shapes, null);

        // Assert
        Assert.result.IsValid, True);
    }

    [Fact]
    public void Registry_RoutesToCorrectHandler_Conv1D()
    {
        // Arrange
        var shapes = new[]
        {
            new long[] { 32, 3, 100 },
            new long[] { 64, 3, 5 }
        };

        // Act
        var handler = _registry.GetHandler(OperationType.Conv1D);
        var result = handler.Validate(shapes, null);

        // Assert
        Assert.result.IsValid, True);
    }

    [Fact]
    public void Registry_RoutesToCorrectHandler_Concat()
    {
        // Arrange
        var shapes = new[]
        {
            new long[] { 32, 128 },
            new long[] { 32, 256 }
        };
        var parameters = new Dictionary<string, object> { { "axis", 1 } };

        // Act
        var handler = _registry.GetHandler(OperationType.Concat);
        var result = handler.Validate(shapes, parameters);

        // Assert
        Assert.result.IsValid, True);
    }

    [Fact]
    public void Registry_RoutesToCorrectHandler_MaxPool2D()
    {
        // Arrange
        var shapes = new[] { new long[] { 32, 64, 112, 112 } };
        var parameters = new Dictionary<string, object> { { "kernel_size", 2 }, { "stride", 2 } };

        // Act
        var handler = _registry.GetHandler(OperationType.MaxPool2D);
        var result = handler.Validate(shapes, parameters);

        // Assert
        Assert.result.IsValid, True);
    }

    [Fact]
    public void Registry_RoutesToCorrectHandler_AveragePool2D()
    {
        // Arrange
        var shapes = new[] { new long[] { 32, 64, 112, 112 } };
        var parameters = new Dictionary<string, object> { { "kernel_size", 2 }, { "stride", 2 } };

        // Act
        var handler = _registry.GetHandler(OperationType.AveragePool2D);
        var result = handler.Validate(shapes, parameters);

        // Assert
        Assert.result.IsValid, True);
    }

    #endregion

    #region Custom Handler Registration Tests

    [Fact]
    public void Registry_CustomHandlerRegistration_OverridesDefault()
    {
        // Arrange
        var customHandler = new MatrixMultiplyDiagnosticsHandler();

        // Act
        _registry.RegisterDiagnosticsHandler(OperationType.MatrixMultiply, customHandler);
        var retrievedHandler = _registry.GetHandler(OperationType.MatrixMultiply);

        // Assert
        Assert.retrievedHandler, Same(customHandler));
    }

    [Fact]
    public void Registry_HasHandler_ReturnsCorrectValue()
    {
        // Act & Assert
        Assert._registry.HasHandler(OperationType.MatrixMultiply), True);
        Assert._registry.HasHandler(OperationType.Conv2D), True);
        Assert._registry.HasHandler(OperationType.Conv1D), True);
        Assert._registry.HasHandler(OperationType.Concat), True);
        Assert._registry.HasHandler(OperationType.MaxPool2D), True);
        Assert._registry.HasHandler(OperationType.AveragePool2D), True);
        Assert._registry.HasHandler(OperationType.Transpose), False);
    }

    #endregion

    #region Error Message Generation Tests

    [Fact]
    public void Handler_GenerateErrorMessage_MatrixMismatch_ContainsDetails()
    {
        // Arrange
        var handler = _registry.GetHandler(OperationType.MatrixMultiply);
        var shapes = new[] { new long[] { 32, 128 }, new long[] { 256, 64 } };

        // Act
        var message = handler.GenerateErrorMessage(shapes, null, "test_layer");

        // Assert
        Assert.message, Contains("Matrix multiplication failed in layer 'test_layer'"));
        Assert.message, Contains("Input shape:"));
        Assert.message, Contains("Weight shape:"));
        Assert.message, Contains("Problem:"));
    }

    [Fact]
    public void Handler_GenerateErrorMessage_Conv2DContainsCalculations()
    {
        // Arrange
        var handler = _registry.GetHandler(OperationType.Conv2D);
        var shapes = new[]
        {
            new long[] { 32, 3, 224, 224 },
            new long[] { 64, 3, 7, 7 }
        };

        // Act
        var message = handler.GenerateErrorMessage(shapes, null, "conv_layer");

        // Assert
        Assert.message, Contains("Conv2D failed in layer 'conv_layer'"));
        Assert.message, Contains("Calculation:"));
        Assert.message, Contains("Output height ="));
        Assert.message, Contains("Output width ="));
    }

    #endregion

    #region Suggestion Generation Tests

    [Fact]
    public void Handler_GenerateSuggestions_MatrixMultiply_HasHelpfulSuggestions()
    {
        // Arrange
        var handler = _registry.GetHandler(OperationType.MatrixMultiply);
        var shapes = new[] { new long[] { 32, 128 }, new long[] { 256, 64 } };

        // Act
        var suggestions = handler.GenerateSuggestions(shapes, null);

        // Assert
        Assert.suggestions, NotEmpty);
        Assert.suggestions.Any(s => s.Contains("Transpose")), True);
    }

    [Fact]
    public void Handler_GenerateSuggestions_Conv2D_HasHelpfulSuggestions()
    {
        // Arrange
        var handler = _registry.GetHandler(OperationType.Conv2D);
        var shapes = new[]
        {
            new long[] { 32, 3, 224, 224 },
            new long[] { 64, 256, 7, 7 }
        };

        // Act
        var suggestions = handler.GenerateSuggestions(shapes, null);

        // Assert
        Assert.suggestions, NotEmpty);
        Assert.suggestions.Any(s => s.Contains("NCHW")), True);
        Assert.suggestions.Any(s => s.Contains("kernel shape")), True);
    }

    #endregion

    #region Issue Detection Tests

    [Fact]
    public void Handler_DetectIssues_MatrixMultiply_FindsCommonProblems()
    {
        // Arrange
        var handler = _registry.GetHandler(OperationType.MatrixMultiply);
        var shapes = new[] { new long[] { 32, 128 }, new long[] { 64, 128 } };

        // Act
        var issues = handler.DetectIssues(shapes, null);

        // Assert
        Assert.issues.GreaterThan, GreaterThan(0));
    }

    [Fact]
    public void Handler_DetectIssues_Conv2D_FindsNHWCConfusion()
    {
        // Arrange
        var handler = _registry.GetHandler(OperationType.Conv2D);
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
}
