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
/// Unit tests for ErrorReportingService.
/// </summary>
public class ErrorReportingServiceTests
{
    [Fact]
    public void CaptureContext_StoresContextCorrectly()
    {
        // Arrange
        var service = new ErrorReportingService();
        var context = new OperationExecutionContext
        {
            LayerName = "test_layer",
            OperationType = OperationType.MatrixMultiply,
            InputTensors = new Tensor[]
            {
                new Tensor(new float[4], new[] { 2, 2 })
            },
            OperationParameters = new Dictionary<string, object> { { "param1", "value1" } }
        };

        // Act
        service.CaptureContext(context);

        // Assert
        var retrieved = service.GetCurrentContext();
        Assert.NotNull(retrieved);
        Assert.Equal("test_layer", retrieved.LayerName);
        Assert.Equal(OperationType.MatrixMultiply, retrieved.OperationType);
        Assert.Equal(1, retrieved.InputTensors.Length);
    }

    [Fact]
    public void ClearContext_RemovesContext()
    {
        // Arrange
        var service = new ErrorReportingService();
        var context = new OperationExecutionContext
        {
            LayerName = "test_layer",
            OperationType = OperationType.Conv2D
        };
        service.CaptureContext(context);

        // Act
        service.ClearContext();

        // Assert
        var retrieved = service.GetCurrentContext();
        Assert.Null(retrieved);
    }

    [Fact]
    public void GenerateShapeMismatchException_ThrowsWithoutContext()
    {
        // Arrange
        var service = new ErrorReportingService();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
        {
            service.GenerateShapeMismatchException("Test error");
        });
    }

    [Fact]
    public void GenerateShapeMismatchException_CreatesValidException()
    {
        // Arrange
        var service = new ErrorReportingService();
        var context = new OperationExecutionContext
        {
            LayerName = "test_layer",
            OperationType = OperationType.MatrixMultiply,
            InputTensors = new Tensor[]
            {
                new Tensor(new float[6], new[] { 2, 3 }),
                new Tensor(new float[6], new[] { 3, 2 })
            },
            OperationParameters = new Dictionary<string, object>(),
            PreviousLayerName = "previous_layer"
        };
        service.CaptureContext(context);

        // Act
        var exception = service.GenerateShapeMismatchException("Shape mismatch test");

        // Assert
        Assert.NotNull(exception);
        Assert.Equal("test_layer", exception.LayerName);
        Assert.Equal(OperationType.MatrixMultiply, exception.OperationType);
        Assert.NotNull(exception.ProblemDescription);
        Assert.NotNull(exception.InputShapes);
        Assert.Equal(2, exception.InputShapes.Count);
    }

    [Fact]
    public void GenerateShapeMismatchException_WithInvalidShapes_CreatesExceptionWithErrors()
    {
        // Arrange
        var service = new ErrorReportingService();
        var context = new OperationExecutionContext
        {
            LayerName = "test_layer",
            OperationType = OperationType.MatrixMultiply,
            InputTensors = new Tensor[]
            {
                new Tensor(new float[6], new[] { 2, 3 }),
                new Tensor(new float[6], new[] { 2, 3 }) // Wrong shape for matrix multiply
            },
            OperationParameters = new Dictionary<string, object>()
        };

        // Enable diagnostics
        MLFrameworkDiagnostics.EnableDiagnostics();

        service.CaptureContext(context);

        // Act
        var exception = service.GenerateShapeMismatchException("Inner dimension mismatch");

        // Assert
        Assert.NotNull(exception);
        Assert.Equal("test_layer", exception.LayerName);
        Assert.NotNull(exception.ProblemDescription);
        Assert.NotNull(exception.InputShapes);
        Assert.True(exception.SuggestedFixes.Count > 0);

        // Cleanup
        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    [Fact]
    public void LogShapeMismatch_LogsWhenVerbose()
    {
        // Arrange
        var service = new ErrorReportingService();
        var operationType = OperationType.Conv2D;
        var inputShapes = new long[][]
        {
            new[] { 1L, 3L, 32L, 32L },
            new[] { 16L, 3L, 3L, 3L }
        };
        var message = "Test shape mismatch message";

        // Enable verbose diagnostics
        MLFrameworkDiagnostics.EnableDiagnostics(verbose: true);

        // Act
        // Note: This will log to console, we just verify it doesn't throw
        var ex = Record.Exception(() =>
        {
            service.LogShapeMismatch(operationType, inputShapes, message);
        });

        // Assert
        Assert.Null(ex);

        // Cleanup
        MLFrameworkDiagnostics.DisableDiagnostics();
    }

    [Fact]
    public void CaptureContext_SetsTimestamp()
    {
        // Arrange
        var service = new ErrorReportingService();
        var before = DateTime.UtcNow.AddSeconds(-1);
        var context = new OperationExecutionContext
        {
            LayerName = "test_layer",
            OperationType = OperationType.Conv2D
        };

        // Act
        service.CaptureContext(context);
        var after = DateTime.UtcNow.AddSeconds(1);

        // Assert
        var retrieved = service.GetCurrentContext();
        Assert.NotNull(retrieved);
        Assert.True(retrieved.Timestamp >= before);
        Assert.True(retrieved.Timestamp <= after);
    }

    [Fact]
    public async Task MultipleThreads_HaveIndependentContexts()
    {
        // Arrange
        var service = new ErrorReportingService();
        var results = new Dictionary<string, string>();

        // Act
        var task1 = Task.Run(() =>
        {
            service.CaptureContext(new OperationExecutionContext { LayerName = "thread1" });
            Task.Delay(50).Wait();
            var context = service.GetCurrentContext();
            lock (results)
            {
                results["thread1"] = context?.LayerName;
            }
        });

        var task2 = Task.Run(() =>
        {
            Task.Delay(25).Wait();
            service.CaptureContext(new OperationExecutionContext { LayerName = "thread2" });
            var context = service.GetCurrentContext();
            lock (results)
            {
                results["thread2"] = context?.LayerName;
            }
        });

        await Task.WhenAll(task1, task2);

        // Assert
        Assert.Equal("thread1", results["thread1"]);
        Assert.Equal("thread2", results["thread2"]);
    }

    [Fact]
    public void GetInputShapes_ReturnsCorrectShapes()
    {
        // Arrange
        var context = new OperationExecutionContext
        {
            InputTensors = new Tensor[]
            {
                new Tensor(new float[4], new[] { 2, 2 }),
                new Tensor(new float[6], new[] { 2, 3 })
            }
        };

        // Act
        var shapes = context.GetInputShapes();

        // Assert
        Assert.NotNull(shapes);
        Assert.Equal(2, shapes.Length);
        Assert.Equal(new[] { 2L, 2L }, shapes[0]);
        Assert.Equal(new[] { 2L, 3L }, shapes[1]);
    }

    [Fact]
    public void GetPreviousLayerShape_ReturnsCorrectShape()
    {
        // Arrange
        var context = new OperationExecutionContext
        {
            PreviousLayerOutput = new Tensor(new float[6], new[] { 2, 3 })
        };

        // Act
        var shape = context.GetPreviousLayerShape();

        // Assert
        Assert.NotNull(shape);
        Assert.Equal(new[] { 2L, 3L }, shape);
    }

    [Fact]
    public void GetPreviousLayerShape_WhenNull_ReturnsNull()
    {
        // Arrange
        var context = new OperationExecutionContext
        {
            PreviousLayerOutput = null
        };

        // Act
        var shape = context.GetPreviousLayerShape();

        // Assert
        Assert.Null(shape);
    }

    [Fact]
    public void GetInputShapes_WhenEmpty_ReturnsEmpty()
    {
        // Arrange
        var context = new OperationExecutionContext
        {
            InputTensors = Array.Empty<Tensor>()
        };

        // Act
        var shapes = context.GetInputShapes();

        // Assert
        Assert.NotNull(shapes);
        Assert.Empty(shapes);
    }

    [Fact]
    public void CaptureContext_WithNullInputTensors_DoesNotThrow()
    {
        // Arrange
        var service = new ErrorReportingService();
        var context = new OperationExecutionContext
        {
            LayerName = "test_layer",
            InputTensors = null
        };

        // Act & Assert
        var ex = Record.Exception(() => service.CaptureContext(context));
        Assert.Null(ex);
    }

    [Fact]
    public void GenerateShapeMismatchException_IncludesSuggestedFixes()
    {
        // Arrange
        var service = new ErrorReportingService();
        var context = new OperationExecutionContext
        {
            LayerName = "test_layer",
            OperationType = OperationType.MatrixMultiply,
            InputTensors = new Tensor[]
            {
                new Tensor(new float[4], new[] { 2, 2 }),
                new Tensor(new float[6], new[] { 3, 2 })
            }
        };

        // Enable diagnostics
        MLFrameworkDiagnostics.EnableDiagnostics();
        service.CaptureContext(context);

        // Act
        var exception = service.GenerateShapeMismatchException("Dimension mismatch");

        // Assert
        Assert.NotNull(exception);
        Assert.NotNull(exception.SuggestedFixes);
        Assert.True(exception.SuggestedFixes.Count > 0);

        // Cleanup
        MLFrameworkDiagnostics.DisableDiagnostics();
    }
}
