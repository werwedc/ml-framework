namespace MLFramework.Communication.Async;

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

/// <summary>
/// Event-based async communication handler
/// </summary>
public class AsyncCommunicationEventHandler : IDisposable
{
    private readonly IAsyncCommunicationBackend _backend;
    private readonly Dictionary<int, List<Action<ICommunicationHandle>>> _eventHandlers;
    private readonly object _lock;
    private bool _disposed;

    /// <summary>
    /// Event fired when any operation completes
    /// </summary>
    public event Action<ICommunicationHandle>? OnOperationComplete;

    /// <summary>
    /// Event fired when an operation fails
    /// </summary>
    public event Action<ICommunicationHandle, Exception>? OnOperationError;

    public AsyncCommunicationEventHandler(IAsyncCommunicationBackend backend)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _eventHandlers = new Dictionary<int, List<Action<ICommunicationHandle>>>();
        _lock = new object();
    }

    /// <summary>
    /// Start async operation and register completion handler
    /// </summary>
    public void StartOperation(
        Func<ICommunicationHandle> startFunc,
        int operationId,
        Action<ICommunicationHandle>? onComplete = null)
    {
        var handle = startFunc();

        lock (_lock)
        {
            if (!_eventHandlers.ContainsKey(operationId))
            {
                _eventHandlers[operationId] = new List<Action<ICommunicationHandle>>();
            }

            if (onComplete != null)
            {
                _eventHandlers[operationId].Add(onComplete);
            }
        }

        // Start monitoring task
        Task.Run(() => MonitorOperation(handle, operationId));
    }

    private async Task MonitorOperation(ICommunicationHandle handle, int operationId)
    {
        try
        {
            await Task.Run(() => handle.Wait());

            // Fire completion events
            OnOperationComplete?.Invoke(handle);

            lock (_lock)
            {
                if (_eventHandlers.ContainsKey(operationId))
                {
                    foreach (var handler in _eventHandlers[operationId])
                    {
                        handler(handle);
                    }
                    _eventHandlers.Remove(operationId);
                }
            }
        }
        catch (Exception ex)
        {
            OnOperationError?.Invoke(handle, ex);
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            lock (_lock)
            {
                _eventHandlers.Clear();
            }
            _disposed = true;
        }
    }
}
