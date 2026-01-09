using System;
using System.Collections.Generic;

namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Interface for error reporting service that captures operation context
/// and generates detailed shape mismatch exceptions.
/// </summary>
public interface IErrorReportingService
{
    /// <summary>
    /// Captures the current operation context.
    /// </summary>
    /// <param name="context">The operation context to capture.</param>
    void CaptureContext(OperationExecutionContext context);

    /// <summary>
    /// Gets the current operation context.
    /// </summary>
    /// <returns>The current operation context, or null if none is set.</returns>
    OperationExecutionContext GetCurrentContext();

    /// <summary>
    /// Clears the current operation context.
    /// </summary>
    void ClearContext();

    /// <summary>
    /// Generates a ShapeMismatchException from the current context.
    /// </summary>
    /// <param name="problemDescription">Description of the problem that occurred.</param>
    /// <returns>A detailed ShapeMismatchException with all context information.</returns>
    ShapeMismatchException GenerateShapeMismatchException(string problemDescription);

    /// <summary>
    /// Logs a shape mismatch error (optional, for debugging purposes).
    /// </summary>
    /// <param name="operationType">The type of operation.</param>
    /// <param name="inputShapes">The input shapes that caused the mismatch.</param>
    /// <param name="message">The message to log.</param>
    void LogShapeMismatch(OperationType operationType, long[][] inputShapes, string message);
}
