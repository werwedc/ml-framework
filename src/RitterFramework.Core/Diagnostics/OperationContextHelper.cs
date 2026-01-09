using System;
using System.Collections.Generic;

namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Helper class for executing operations with automatic error context capture.
/// Provides a wrapper to track operation context and generate detailed exceptions.
/// </summary>
public static class OperationContextHelper
{
    /// <summary>
    /// Executes an operation with automatic context capture and error reporting.
    /// </summary>
    /// <typeparam name="T">The return type of the operation.</typeparam>
    /// <param name="layerName">Name of the layer/module executing the operation.</param>
    /// <param name="operationType">The type of operation being performed.</param>
    /// <param name="inputs">Input tensors for the operation.</param>
    /// <param name="parameters">Optional operation parameters (e.g., stride, padding).</param>
    /// <param name="operation">The operation delegate to execute.</param>
    /// <returns>The result of the operation.</returns>
    /// <exception cref="ShapeMismatchException">Thrown when a shape mismatch occurs.</exception>
    public static T ExecuteWithContext<T>(
        string layerName,
        OperationType operationType,
        global::RitterFramework.Core.Tensor.Tensor[] inputs,
        IDictionary<string, object> parameters,
        Func<T> operation)
    {
        // Create context
        var context = new OperationExecutionContext
        {
            LayerName = layerName ?? "unknown",
            OperationType = operationType,
            InputTensors = inputs ?? Array.Empty<global::RitterFramework.Core.Tensor.Tensor>(),
            OperationParameters = parameters ?? new Dictionary<string, object>()
        };

        // Get error reporting service
        var service = GetErrorReportingService();

        // Capture context
        service.CaptureContext(context);

        try
        {
            return operation();
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains("shape", StringComparison.OrdinalIgnoreCase))
        {
            // Generate detailed shape mismatch exception
            throw service.GenerateShapeMismatchException(ex.Message);
        }
        catch (ArgumentException ex) when (ex.Message.Contains("shape", StringComparison.OrdinalIgnoreCase))
        {
            // Generate detailed shape mismatch exception
            throw service.GenerateShapeMismatchException(ex.Message);
        }
        finally
        {
            // Always clear context to prevent memory leaks
            service.ClearContext();
        }
    }

    /// <summary>
    /// Executes an operation with automatic context capture (without parameters).
    /// </summary>
    /// <typeparam name="T">The return type of the operation.</typeparam>
    /// <param name="layerName">Name of the layer/module executing the operation.</param>
    /// <param name="operationType">The type of operation being performed.</param>
    /// <param name="inputs">Input tensors for the operation.</param>
    /// <param name="operation">The operation delegate to execute.</param>
    /// <returns>The result of the operation.</returns>
    public static T ExecuteWithContext<T>(
        string layerName,
        OperationType operationType,
        global::RitterFramework.Core.Tensor.Tensor[] inputs,
        Func<T> operation)
    {
        return ExecuteWithContext(layerName, operationType, inputs, null, operation);
    }

    /// <summary>
    /// Executes an operation with automatic context capture (single input tensor).
    /// </summary>
    /// <typeparam name="T">The return type of the operation.</typeparam>
    /// <param name="layerName">Name of the layer/module executing the operation.</param>
    /// <param name="operationType">The type of operation being performed.</param>
    /// <param name="input">Single input tensor for the operation.</param>
    /// <param name="operation">The operation delegate to execute.</param>
    /// <returns>The result of the operation.</returns>
    public static T ExecuteWithContext<T>(
        string layerName,
        OperationType operationType,
        global::RitterFramework.Core.Tensor.Tensor input,
        Func<T> operation)
    {
        return ExecuteWithContext(layerName, operationType, new[] { input }, operation);
    }

    /// <summary>
    /// Executes an operation with automatic context capture (single input with parameters).
    /// </summary>
    /// <typeparam name="T">The return type of the operation.</typeparam>
    /// <param name="layerName">Name of the layer/module executing the operation.</param>
    /// <param name="operationType">The type of operation being performed.</param>
    /// <param name="input">Single input tensor for the operation.</param>
    /// <param name="parameters">Optional operation parameters.</param>
    /// <param name="operation">The operation delegate to execute.</param>
    /// <returns>The result of the operation.</returns>
    public static T ExecuteWithContext<T>(
        string layerName,
        OperationType operationType,
        global::RitterFramework.Core.Tensor.Tensor input,
        IDictionary<string, object> parameters,
        Func<T> operation)
    {
        return ExecuteWithContext(layerName, operationType, new[] { input }, parameters, operation);
    }

    /// <summary>
    /// Gets the error reporting service instance.
    /// In a production environment, this would use dependency injection.
    /// </summary>
    public static IErrorReportingService GetErrorReportingService()
    {
        // For now, use a static instance
        // TODO: Replace with proper DI when available
        return _errorReportingService ??= new ErrorReportingService();
    }

    private static IErrorReportingService _errorReportingService;
}
