namespace MLFramework.Communication.Operations;

using RitterFramework.Core.Tensor;
using System.Threading.Tasks;

/// <summary>
/// Pending operation handle for async operations
/// </summary>
public class PendingOperationHandle : ICommunicationHandle
{
    private readonly Task<Tensor> _task;

    public PendingOperationHandle(Task<Tensor> task)
    {
        _task = task ?? throw new ArgumentNullException(nameof(task));
    }

    public bool IsCompleted => _task.IsCompleted;

    public void Wait()
    {
        _task.Wait();
    }

    public bool TryWait(int timeoutMs)
    {
        return _task.Wait(timeoutMs);
    }

    public Tensor GetResult()
    {
        if (!IsCompleted)
        {
            throw new InvalidOperationException("Operation has not completed yet");
        }

        return _task.Result;
    }
}
