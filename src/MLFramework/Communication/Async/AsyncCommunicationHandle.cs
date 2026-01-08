namespace MLFramework.Communication.Async;

using System.Threading;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;

/// <summary>
/// Enhanced communication handle for async operations
/// </summary>
public class AsyncCommunicationHandle : ICommunicationHandle
{
    private readonly Task<Tensor> _task;
    private readonly CancellationTokenSource _cts;
    private bool _completed;
    private Tensor? _result;

    public AsyncCommunicationHandle(Task<Tensor> task, CancellationTokenSource? cts = null)
    {
        _task = task ?? throw new ArgumentNullException(nameof(task));
        _cts = cts ?? new CancellationTokenSource();
    }

    public bool IsCompleted
    {
        get
        {
            if (_completed) return true;
            _completed = _task.IsCompleted;
            return _completed;
        }
    }

    public void Wait()
    {
        _task.Wait();
        _result = _task.Result;
    }

    public bool TryWait(int timeoutMs)
    {
        if (_task.Wait(timeoutMs))
        {
            _result = _task.Result;
            return true;
        }
        return false;
    }

    public Tensor GetResult()
    {
        if (!_completed)
        {
            throw new InvalidOperationException("Operation has not completed yet");
        }

        if (_result == null)
        {
            throw new InvalidOperationException("Result is not available");
        }

        return _result;
    }

    /// <summary>
    /// Get result as Task for async/await pattern
    /// </summary>
    public Task<Tensor> AsTask()
    {
        return _task;
    }

    /// <summary>
    /// Cancel the operation if possible
    /// </summary>
    public void Cancel()
    {
        _cts.Cancel();
    }

    /// <summary>
    /// Check if the operation was cancelled
    /// </summary>
    public bool IsCancelled
    {
        get { return _cts.IsCancellationRequested; }
    }
}
