# Technical Spec: Error Reporting Core

## Overview
Implement the core error reporting functionality that captures operation context at error sites and generates basic descriptive error messages. This integrates the diagnostics system with the existing tensor operations.

## Requirements

### Error Context Capture
Create a system to capture context when errors occur:

```csharp
public class OperationExecutionContext
{
    public string LayerName { get; set; }
    public OperationType OperationType { get; set; }
    public Tensor[] InputTensors { get; set; }
    public IDictionary<string, object> OperationParameters { get; set; }
    public string PreviousLayerName { get; set; }
    public Tensor PreviousLayerOutput { get; set; }
    public DateTime Timestamp { get; set; }
}
```

### Error Reporting Service
```csharp
public interface IErrorReportingService
{
    // Capture current operation context
    void CaptureContext(OperationExecutionContext context);

    // Get current context
    OperationExecutionContext GetCurrentContext();

    // Clear context
    void ClearContext();

    // Generate ShapeMismatchException from current context
    ShapeMismatchException GenerateShapeMismatchException(string problemDescription);

    // Log shape mismatch (optional, for debugging)
    void LogShapeMismatch(OperationType operationType, long[][] inputShapes, string message);
}
```

### Default Implementation
```csharp
public class ErrorReportingService : IErrorReportingService
{
    private static AsyncLocal<OperationExecutionContext> _currentContext =
        new AsyncLocal<OperationExecutionContext>();

    public void CaptureContext(OperationExecutionContext context)
    {
        context.Timestamp = DateTime.UtcNow;
        _currentContext.Value = context;
    }

    public OperationExecutionContext GetCurrentContext()
    {
        return _currentContext.Value;
    }

    public void ClearContext()
    {
        _currentContext.Value = null;
    }

    public ShapeMismatchException GenerateShapeMismatchException(string problemDescription)
    {
        var context = _currentContext.Value;

        if (context == null)
        {
            throw new InvalidOperationException("No operation context available");
        }

        var inputShapes = context.InputTensors.Select(t => t.Shape.ToArray()).ToArray();
        var diagnostics = MLFrameworkDiagnostics.GetShapeDiagnostics(
            context.OperationType,
            context.InputTensors,
            context.LayerName,
            context.OperationParameters);

        var suggestedFixes = MLFrameworkDiagnostics.GenerateSuggestedFixes(diagnostics);

        var exception = new ShapeMismatchException(
            context.LayerName,
            context.OperationType,
            inputShapes,
            diagnostics.ExpectedShapes,
            problemDescription,
            suggestedFixes,
            context.InputTensors[0]?.Shape[0], // Batch size if applicable
            context.PreviousLayerName);

        return exception;
    }

    public void LogShapeMismatch(OperationType operationType, long[][] inputShapes, string message)
    {
        if (MLFrameworkDiagnostics.IsVerbose)
        {
            var shapes = string.Join(", ", inputShapes.Select(s => $"[{string.Join(", ", s)}]"));
            Console.WriteLine($"[SHAPE MISMATCH] {operationType}: {shapes} - {message}");
        }
    }
}
```

### Operation Wrapper for Error Context
Create a wrapper/helper to automatically capture context:

```csharp
public static class OperationContextHelper
{
    public static T ExecuteWithContext<T>(
        string layerName,
        OperationType operationType,
        Tensor[] inputs,
        IDictionary<string, object> parameters,
        Func<T> operation)
    {
        var context = new OperationExecutionContext
        {
            LayerName = layerName,
            OperationType = operationType,
            InputTensors = inputs,
            OperationParameters = parameters
        };

        var service = ServiceLocator.Resolve<IErrorReportingService>();
        service.CaptureContext(context);

        try
        {
            return operation();
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("shape", StringComparison.OrdinalIgnoreCase))
        {
            throw service.GenerateShapeMismatchException(ex.Message);
        }
        finally
        {
            service.ClearContext();
        }
    }
}
```

### Integration with Tensor Operations
Extend existing tensor operations to use the error reporting system:

#### Example: MatrixMultiply with Error Reporting
```csharp
public static class TensorOperations
{
    public static Tensor MatrixMultiply(
        this Tensor left,
        Tensor right,
        string layerName = null,
        IDictionary<string, object> parameters = null)
    {
        return OperationContextHelper.ExecuteWithContext(
            layerName ?? "matrix_multiply",
            OperationType.MatrixMultiply,
            new[] { left, right },
            parameters,
            () =>
            {
                // Existing matrix multiply implementation
                ValidateShapes(left, right);
                return PerformMatrixMultiply(left, right);
            });
    }

    private static void ValidateShapes(Tensor left, Tensor right)
    {
        // Check if diagnostics are enabled
        if (MLFrameworkDiagnostics.IsEnabled)
        {
            var isValid = MLFrameworkDiagnostics.CheckShapes(
                OperationType.MatrixMultiply,
                new[] { left, right });

            if (!isValid)
            {
                // Get diagnostics and generate exception
                var diagnostics = MLFrameworkDiagnostics.GetShapeDiagnostics(
                    OperationType.MatrixMultiply,
                    new[] { left, right });

                var problemDescription = diagnostics.Errors.Count > 0
                    ? diagnostics.Errors[0]
                    : "Matrix multiplication shape mismatch";

                throw new InvalidOperationException(problemDescription);
            }
        }
        else
        {
            // Basic validation for when diagnostics are disabled
            if (left.Shape.Length < 2 || right.Shape.Length < 2)
            {
                throw new InvalidOperationException("Matrices must have at least 2 dimensions");
            }

            if (left.Shape[left.Shape.Length - 1] != right.Shape[right.Shape.Length - 2])
            {
                throw new InvalidOperationException(
                    $"Cannot multiply matrices: inner dimensions mismatch " +
                    $"({left.Shape[left.Shape.Length - 1]} != {right.Shape[right.Shape.Length - 2]})");
            }
        }
    }
}
```

### Layer Context Tracking
Add context tracking to Module/Layer classes:

```csharp
public abstract class Module
{
    protected string Name { get; set; }

    protected Tensor ForwardWithContext(
        Tensor input,
        OperationType operationType,
        IDictionary<string, object> parameters = null)
    {
        var context = new OperationExecutionContext
        {
            LayerName = Name,
            OperationType = operationType,
            InputTensors = new[] { input },
            OperationParameters = parameters,
            PreviousLayerName = GetPreviousLayerName()
        };

        var service = ServiceLocator.Resolve<IErrorReportingService>();
        service.CaptureContext(context);

        try
        {
            return Forward(input);
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("shape", StringComparison.OrdinalIgnoreCase))
        {
            throw service.GenerateShapeMismatchException(ex.Message);
        }
        finally
        {
            service.ClearContext();
        }
    }

    protected abstract Tensor Forward(Tensor input);

    private string GetPreviousLayerName()
    {
        // Implementation depends on model structure
        return null;
    }
}
```

## Deliverables
- File: `src/Diagnostics/OperationExecutionContext.cs`
- File: `src/Diagnostics/IErrorReportingService.cs`
- File: `src/Diagnostics/ErrorReportingService.cs`
- File: `src/Diagnostics/OperationContextHelper.cs`
- File: `src/Operations/TensorOperationsExtensions.cs` (add error reporting to existing operations)
- File: `src/Modules/ModuleExtensions.cs` (add context tracking to Module base)

## Testing Requirements
Create unit tests in `tests/Diagnostics/ErrorReportingServiceTests.cs`:
- Test context capture and retrieval
- Test context clearing
- Test exception generation from context
- Test shape mismatch logging in verbose mode
- Test that context is isolated between threads (AsyncLocal behavior)

Create integration tests in `tests/Integration/OperationErrorReportingTests.cs`:
- Test MatrixMultiply with invalid shapes throws ShapeMismatchException
- Test Conv2D with invalid shapes throws ShapeMismatchException
- Test that layer names are captured correctly
- Test that previous layer context is preserved
- Test that diagnostics can be disabled
- Test with multiple layers in a model

## Notes
- Use AsyncLocal for thread-safe context tracking
- Ensure proper cleanup in finally blocks
- Provide extension methods for easy integration with existing code
- Make error reporting opt-in to avoid performance overhead
- Consider using dependency injection for ErrorReportingService
