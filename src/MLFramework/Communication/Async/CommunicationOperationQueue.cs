namespace MLFramework.Communication.Async;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;

/// <summary>
/// Queue for managing multiple async communication operations
/// </summary>
public class CommunicationOperationQueue : IDisposable
{
    private readonly List<ICommunicationHandle> _operations;
    private readonly object _lock;
    private bool _disposed;

    public int PendingOperationsCount
    {
        get
        {
            lock (_lock)
            {
                return _operations.Count(o => !o.IsCompleted);
            }
        }
    }

    public CommunicationOperationQueue()
    {
        _operations = new List<ICommunicationHandle>();
        _lock = new object();
    }

    /// <summary>
    /// Add an operation to the queue
    /// </summary>
    public void Enqueue(ICommunicationHandle handle)
    {
        if (handle == null)
            throw new ArgumentNullException(nameof(handle));

        lock (_lock)
        {
            _operations.Add(handle);
        }
    }

    /// <summary>
    /// Wait for all operations to complete
    /// </summary>
    public void WaitForAll()
    {
        lock (_lock)
        {
            foreach (var operation in _operations)
            {
                operation.Wait();
            }
        }
    }

    /// <summary>
    /// Wait for all operations with timeout
    /// </summary>
    /// <returns>True if all completed, false if timeout</returns>
    public bool TryWaitForAll(int timeoutMs)
    {
        var stopwatch = Stopwatch.StartNew();
        lock (_lock)
        {
            foreach (var operation in _operations)
            {
                var remaining = (int)(timeoutMs - stopwatch.ElapsedMilliseconds);
                if (remaining <= 0)
                {
                    return false;
                }

                if (!operation.TryWait(remaining))
                {
                    return false;
                }
            }
        }
        return true;
    }

    /// <summary>
    /// Wait for any operation to complete
    /// </summary>
    /// <returns>Index of completed operation or -1 if timeout</returns>
    public int WaitForAny(int timeoutMs = -1)
    {
        lock (_lock)
        {
            for (int i = 0; i < _operations.Count; i++)
            {
                if (_operations[i].IsCompleted)
                {
                    return i;
                }
            }

            // Poll for completion
            var stopwatch = Stopwatch.StartNew();
            while (timeoutMs == -1 || stopwatch.ElapsedMilliseconds < timeoutMs)
            {
                for (int i = 0; i < _operations.Count; i++)
                {
                    if (_operations[i].IsCompleted)
                    {
                        return i;
                    }
                }
                Thread.Sleep(1);
            }

            return -1;
        }
    }

    /// <summary>
    /// Clear all completed operations
    /// </summary>
    public void ClearCompleted()
    {
        lock (_lock)
        {
            _operations.RemoveAll(o => o.IsCompleted);
        }
    }

    /// <summary>
    /// Cancel all pending operations
    /// </summary>
    public void CancelAll()
    {
        lock (_lock)
        {
            foreach (var operation in _operations)
            {
                if (operation is AsyncCommunicationHandle asyncHandle)
                {
                    asyncHandle.Cancel();
                }
            }
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            CancelAll();
            lock (_lock)
            {
                _operations.Clear();
            }
            _disposed = true;
        }
    }
}
