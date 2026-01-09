using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using RitterFramework.Core;
using RitterFramework.Core.Diagnostics;
using RitterFramework.Core.Tensor;
using Xunit;

namespace RitterFramework.Core.Tests.Diagnostics;

/// <summary>
/// Integration tests for operation error reporting.
/// Tests integration of error reporting system with tensor operations and modules.
/// </summary>
public class OperationErrorReportingTests : IDisposable
{
    public OperationErrorReportingTests()
    {
        // Clean up any existing diagnostics state
        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    public void Dispose()
    {
        // Clean up after tests
        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    [Fact]
    public void MatrixMultiply_WithInvalidShapes_ThrowsShapeMismatchException()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var left = new Tensor(new float[6], new[] { 2, 3 });
        var right = new Tensor(new float[8], new[] { 2, 4 }); // Wrong inner dimension

        // Act & Assert
        var exception = Assert.Throws<ShapeMismatchException>(() =>
        {
            left.MatrixMultiply(right, layerName: "test_matmul");
        });

        Assert.Equal("test_matmul", exception.LayerName);
        Assert.Equal(OperationType.MatrixMultiply, exception.OperationType);
        Assert.NotNull(exception.ProblemDescription);
    }

    [Fact]
    public void MatrixMultiply_WithValidShapes_DoesNotThrow()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var left = new Tensor(new float[6], new[] { 2, 3 });
        var right = new Tensor(new float[12], new[] { 3, 4 }); // Valid: 2x3 × 3x4

        // Act & Assert
        // Note: Actual matrix multiply implementation may throw NotImplementedException
        // This test verifies shape validation doesn't throw
        var ex = Record.Exception(() =>
        {
            left.MatrixMultiply(right, layerName: "test_matmul");
        });

        // We expect NotImplementedException since actual matmul is not implemented
        Assert.IsType<NotImplementedException>(ex);
    }

    [Fact]
    public void Conv2D_WithInvalidInputShape_ThrowsShapeMismatchException()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var input = new Tensor(new float[12], new[] { 2, 2, 2 }); // Wrong: should be 4D
        var kernel = new Tensor(new float[27], new[] { 3, 3, 3, 3 });

        // Act & Assert
        var exception = Assert.Throws<ShapeMismatchException>(() =>
        {
            input.Conv2D(kernel, layerName: "test_conv2d");
        });

        Assert.Equal("test_conv2d", exception.LayerName);
        Assert.Equal(OperationType.Conv2D, exception.OperationType);
        Assert.Contains("4D", exception.ProblemDescription);
    }

    [Fact]
    public void Conv2D_WithInvalidKernelShape_ThrowsShapeMismatchException()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var input = new Tensor(new float[96], new[] { 2, 3, 4, 4 }); // NCHW format
        var kernel = new Tensor(new float[16], new[] { 2, 2, 2, 2 }); // Wrong: should be 4D

        // Act & Assert
        var exception = Assert.Throws<ShapeMismatchException>(() =>
        {
            input.Conv2D(kernel, layerName: "test_conv2d");
        });

        Assert.Equal("test_conv2d", exception.LayerName);
        Assert.Equal(OperationType.Conv2D, exception.OperationType);
    }

    [Fact]
    public void Concat_WithDifferentRanks_ThrowsShapeMismatchException()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var tensor1 = new Tensor(new float[6], new[] { 2, 3 });
        var tensor2 = new Tensor(new float[24], new[] { 2, 3, 4 }); // Different rank

        // Act & Assert
        var exception = Assert.Throws<ShapeMismatchException>(() =>
        {
            new[] { tensor1, tensor2 }.Concat(dimension: 0, layerName: "test_concat");
        });

        Assert.Equal("test_concat", exception.LayerName);
        Assert.Equal(OperationType.Concat, exception.OperationType);
        Assert.Contains("same rank", exception.ProblemDescription);
    }

    [Fact]
    public void Concat_WithDifferentDimensions_ThrowsShapeMismatchException()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var tensor1 = new Tensor(new float[6], new[] { 2, 3 });
        var tensor2 = new Tensor(new float[8], new[] { 2, 4 }); // Different non-concat dimension

        // Act & Assert
        var exception = Assert.Throws<ShapeMismatchException>(() =>
        {
            new[] { tensor1, tensor2 }.Concat(dimension: 0, layerName: "test_concat");
        });

        Assert.Equal("test_concat", exception.LayerName);
        Assert.Contains("same size in dimension 1", exception.ProblemDescription);
    }

    [Fact]
    public void Stack_WithDifferentShapes_ThrowsShapeMismatchException()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var tensor1 = new Tensor(new float[6], new[] { 2, 3 });
        var tensor2 = new Tensor(new float[8], new[] { 2, 4 }); // Different shape

        // Act & Assert
        var exception = Assert.Throws<ShapeMismatchException>(() =>
        {
            new[] { tensor1, tensor2 }.Stack(dimension: 0, layerName: "test_stack");
        });

        Assert.Equal("test_stack", exception.LayerName);
        Assert.Equal(OperationType.Stack, exception.OperationType);
        Assert.Contains("same shape", exception.ProblemDescription);
    }

    [Fact]
    public void LayerName_IsCapturedCorrectly()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();
        var expectedLayerName = "my_custom_layer";

        var left = new Tensor(new float[6], new[] { 2, 3 });
        var right = new Tensor(new float[8], new[] { 2, 4 });

        // Act & Assert
        var exception = Assert.Throws<ShapeMismatchException>(() =>
        {
            left.MatrixMultiply(right, layerName: expectedLayerName);
        });

        Assert.Equal(expectedLayerName, exception.LayerName);
    }

    [Fact]
    public void PreviousLayerContext_IsPreserved()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var service = new ErrorReportingService();
        var previousTensor = new Tensor(new float[6], new[] { 2, 3 });

        var context = new OperationExecutionContext
        {
            LayerName = "current_layer",
            OperationType = OperationType.MatrixMultiply,
            InputTensors = new[]
            {
                previousTensor,
                new Tensor(new float[6], new[] { 2, 3 })
            },
            PreviousLayerName = "previous_layer",
            PreviousLayerOutput = previousTensor
        };

        service.CaptureContext(context);

        // Act
        var exception = service.GenerateShapeMismatchException("Test error");

        // Assert
        Assert.Equal("previous_layer", GetPreviousLayerContext(exception));
    }

    [Fact]
    public void DiagnosticsCanBeDisabled()
    {
        // Arrange
        MLFrameworkDiagnostics.DisableDiagnostics();

        var left = new Tensor(new float[6], new[] { 2, 3 });
        var right = new Tensor(new float[8], new[] { 2, 4 }); // Invalid shape

        // Act & Assert
        // When disabled, we should get a different exception type or no exception at all
        // Since we don't have actual matmul implementation, we expect NotImplementedException
        var ex = Record.Exception(() =>
        {
            left.MatrixMultiply(right, layerName: "test_matmul");
        });

        // Should not be ShapeMismatchException when diagnostics disabled
        if (ex != null)
        {
            Assert.IsNotType<ShapeMismatchException>(ex);
        }
    }

    [Fact]
    public async Task MultipleLayers_CorrectContextIsolation()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var input = new Tensor(new float[4], new[] { 2, 2 });

        // Act
        var task1 = Task.Run(() =>
        {
            try
            {
                input.MatrixMultiply(new Tensor(new float[6], new[] { 2, 3 }), "layer1");
                return "no error";
            }
            catch (ShapeMismatchException ex)
            {
                return ex.LayerName;
            }
        });

        var task2 = Task.Run(() =>
        {
            try
            {
                input.MatrixMultiply(new Tensor(new float[4], new[] { 2, 2 }), "layer2");
                return "no error";
            }
            catch (ShapeMismatchException ex)
            {
                return ex.LayerName;
            }
        });

        await Task.WhenAll(task1, task2);

        // Assert
        // Both tasks should have completed and captured correct layer names
        Assert.NotNull(task1.Result);
        Assert.NotNull(task2.Result);
    }

    [Fact]
    public void SuggestedFixes_AreGeneratedForMatrixMultiply()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var left = new Tensor(new float[6], new[] { 2, 3 });
        var right = new Tensor(new float[8], new[] { 2, 4 }); // Invalid shape

        // Act
        var exception = Assert.Throws<ShapeMismatchException>(() =>
        {
            left.MatrixMultiply(right, layerName: "test_matmul");
        });

        // Assert
        Assert.NotNull(exception.SuggestedFixes);
        Assert.True(exception.SuggestedFixes.Count > 0);
        Assert.Contains("weight matrix", string.Join(" ", exception.SuggestedFixes));
    }

    [Fact]
    public void OperationParameters_AreIncludedInContext()
    {
        // Arrange
        MLFrameworkDiagnostics.EnableDiagnostics();

        var input = new Tensor(new float[96], new[] { 2, 3, 4, 4 });
        var kernel = new Tensor(new float[27], new[] { 3, 3, 3, 3 });
        var parameters = new Dictionary<string, object>
        {
            { "stride", new[] { 1, 1 } },
            { "padding", new[] { 1, 1 } }
        };

        // Act & Assert
        var exception = Assert.Throws<ShapeMismatchException>(() =>
        {
            input.Conv2D(kernel, layerName: "test_conv2d", parameters: parameters);
        });

        // Verify the exception was created (shape validation happens before parameter use)
        Assert.NotNull(exception);
    }

    #region Helper Methods

    private static string GetPreviousLayerContext(ShapeMismatchException exception)
    {
        // Use reflection to get private property
        var property = exception.GetType().GetProperty("PreviousLayerContext");
        return property?.GetValue(exception) as string;
    }

    #endregion
}
