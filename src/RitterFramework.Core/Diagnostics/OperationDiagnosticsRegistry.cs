namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Default implementation of the operation diagnostics registry.
/// Pre-registers handlers for common operations and allows custom registration.
/// </summary>
public class OperationDiagnosticsRegistry : IOperationDiagnosticsRegistry
{
    private readonly Dictionary<OperationType, IOperationDiagnosticsHandler> _handlers;

    /// <summary>
    /// Create a new registry with default handlers registered.
    /// </summary>
    public OperationDiagnosticsRegistry()
    {
        _handlers = new Dictionary<OperationType, IOperationDiagnosticsHandler>();

        // Register default handlers
        RegisterDiagnosticsHandler(OperationType.MatrixMultiply, new MatrixMultiplyDiagnosticsHandler());
        RegisterDiagnosticsHandler(OperationType.Conv2D, new Conv2DDiagnosticsHandler());
        RegisterDiagnosticsHandler(OperationType.Conv1D, new Conv1DDiagnosticsHandler());
        RegisterDiagnosticsHandler(OperationType.Concat, new ConcatDiagnosticsHandler());
        RegisterDiagnosticsHandler(OperationType.MaxPool2D, new PoolingDiagnosticsHandler(OperationType.MaxPool2D));
        RegisterDiagnosticsHandler(OperationType.AveragePool2D, new PoolingDiagnosticsHandler(OperationType.AveragePool2D));
    }

    /// <inheritdoc/>
    public void RegisterDiagnosticsHandler(OperationType operationType, IOperationDiagnosticsHandler handler)
    {
        if (handler == null)
        {
            throw new ArgumentNullException(nameof(handler));
        }

        _handlers[operationType] = handler;
    }

    /// <inheritdoc/>
    public IOperationDiagnosticsHandler GetHandler(OperationType operationType)
    {
        return _handlers.TryGetValue(operationType, out var handler) ? handler : null;
    }

    /// <inheritdoc/>
    public bool HasHandler(OperationType operationType)
    {
        return _handlers.ContainsKey(operationType);
    }
}
