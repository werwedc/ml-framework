namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Registry for operation-specific diagnostic handlers.
/// Allows registration and retrieval of handlers for different operation types.
/// </summary>
public interface IOperationDiagnosticsRegistry
{
    /// <summary>
    /// Register a diagnostics handler for a specific operation type.
    /// </summary>
    /// <param name="operationType">The operation type.</param>
    /// <param name="handler">The diagnostics handler.</param>
    void RegisterDiagnosticsHandler(
        OperationType operationType,
        IOperationDiagnosticsHandler handler);

    /// <summary>
    /// Get the diagnostics handler for a specific operation type.
    /// </summary>
    /// <param name="operationType">The operation type.</param>
    /// <returns>The diagnostics handler, or null if not registered.</returns>
    IOperationDiagnosticsHandler GetHandler(OperationType operationType);

    /// <summary>
    /// Check if a handler is registered for the given operation type.
    /// </summary>
    /// <param name="operationType">The operation type.</param>
    /// <returns>True if a handler is registered, false otherwise.</returns>
    bool HasHandler(OperationType operationType);
}
