using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Default implementation of IErrorReportingService using AsyncLocal for thread-safe context tracking.
/// </summary>
public class ErrorReportingService : IErrorReportingService
{
    private static readonly AsyncLocal<OperationExecutionContext> _currentContext =
        new AsyncLocal<OperationExecutionContext>();

    /// <summary>
    /// Captures the current operation context.
    /// </summary>
    /// <param name="context">The operation context to capture.</param>
    public void CaptureContext(OperationExecutionContext context)
    {
        if (context == null)
        {
            throw new ArgumentNullException(nameof(context));
        }

        context.Timestamp = DateTime.UtcNow;
        _currentContext.Value = context;
    }

    /// <summary>
    /// Gets the current operation context.
    /// </summary>
    /// <returns>The current operation context, or null if none is set.</returns>
    public OperationExecutionContext GetCurrentContext()
    {
        return _currentContext.Value;
    }

    /// <summary>
    /// Clears the current operation context.
    /// </summary>
    public void ClearContext()
    {
        _currentContext.Value = null;
    }

    /// <summary>
    /// Generates a ShapeMismatchException from the current context.
    /// </summary>
    /// <param name="problemDescription">Description of the problem that occurred.</param>
    /// <returns>A detailed ShapeMismatchException with all context information.</returns>
    public ShapeMismatchException GenerateShapeMismatchException(string problemDescription)
    {
        var context = _currentContext.Value;

        if (context == null)
        {
            throw new InvalidOperationException("No operation context available. Cannot generate shape mismatch exception.");
        }

        var inputShapes = context.GetInputShapes();

        // Get diagnostics from MLFrameworkDiagnostics
        var diagnostics = MLFrameworkDiagnostics.GetShapeDiagnostics(
            context.OperationType,
            context.InputTensors,
            context.LayerName,
            context.OperationParameters);

        // Generate suggested fixes
        var suggestedFixes = MLFrameworkDiagnostics.GenerateSuggestedFixes(diagnostics);

        // Get batch size if applicable
        long? batchSize = null;
        if (inputShapes.Length > 0 && inputShapes[0].Length > 0)
        {
            batchSize = inputShapes[0][0];
        }

        // Get previous layer context string
        var previousLayerContext = BuildPreviousLayerContext(context);

        // Create the exception
        var exception = new ShapeMismatchException(
            problemDescription,
            context.LayerName,
            context.OperationType,
            inputShapes.ToList(),
            diagnostics.ExpectedShapes.ToList(),
            problemDescription,
            suggestedFixes);

        // Set additional context properties (using reflection or extension methods)
        // Note: These properties would need to be added to ShapeMismatchException
        SetAdditionalExceptionProperties(exception, context, previousLayerContext, batchSize);

        return exception;
    }

    /// <summary>
    /// Logs a shape mismatch error (optional, for debugging purposes).
    /// </summary>
    /// <param name="operationType">The type of operation.</param>
    /// <param name="inputShapes">The input shapes that caused the mismatch.</param>
    /// <param name="message">The message to log.</param>
    public void LogShapeMismatch(OperationType operationType, long[][] inputShapes, string message)
    {
        if (MLFrameworkDiagnostics.IsVerbose)
        {
            var shapes = string.Join(", ", inputShapes.Select(s => $"[{string.Join(", ", s)}]"));
            Console.WriteLine($"[SHAPE MISMATCH] {operationType}: {shapes} - {message}");
        }
    }

    /// <summary>
    /// Builds a context string describing the previous layer.
    /// </summary>
    private static string BuildPreviousLayerContext(OperationExecutionContext context)
    {
        if (string.IsNullOrEmpty(context.PreviousLayerName))
        {
            return null;
        }

        var contextBuilder = new System.Text.StringBuilder();
        contextBuilder.Append($"Previous layer: {context.PreviousLayerName}");

        if (context.PreviousLayerOutput != null)
        {
            var previousShape = context.PreviousLayerOutput.GetShapeString();
            contextBuilder.Append($" (shape: {previousShape})");
        }

        return contextBuilder.ToString();
    }

    /// <summary>
    /// Sets additional exception properties that may not be in the base ShapeMismatchException.
    /// </summary>
    private static void SetAdditionalExceptionProperties(
        ShapeMismatchException exception,
        OperationExecutionContext context,
        string previousLayerContext,
        long? batchSize)
    {
        // Use reflection to set properties if they exist
        var exceptionType = exception.GetType();

        var previousLayerProperty = exceptionType.GetProperty("PreviousLayerContext");
        if (previousLayerProperty != null && previousLayerProperty.CanWrite)
        {
            previousLayerProperty.SetValue(exception, previousLayerContext);
        }

        var batchSizeProperty = exceptionType.GetProperty("BatchSize");
        if (batchSizeProperty != null && batchSizeProperty.CanWrite)
        {
            batchSizeProperty.SetValue(exception, batchSize);
        }
    }
}
