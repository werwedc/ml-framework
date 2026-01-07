namespace MLFramework.Communication.Operations;

using RitterFramework.Core.Tensor;

/// <summary>
/// Concrete implementation of ICommunicationHandle for synchronous operations
/// </summary>
public class CompletedOperationHandle : ICommunicationHandle
{
    private readonly Tensor _result;

    public CompletedOperationHandle(Tensor result)
    {
        _result = result ?? throw new ArgumentNullException(nameof(result));
    }

    public bool IsCompleted => true;

    public void Wait()
    {
        // No-op for already completed operation
    }

    public bool TryWait(int timeoutMs)
    {
        return true; // Always completed
    }

    public Tensor GetResult()
    {
        return _result;
    }
}
