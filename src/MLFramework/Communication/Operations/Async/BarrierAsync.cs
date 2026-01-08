namespace MLFramework.Communication.Operations.Async;

using MLFramework.Communication;
using MLFramework.Distributed.Communication;

/// <summary>
/// Asynchronous barrier operation
/// </summary>
public static class BarrierAsync
{
    /// <summary>
    /// Barrier asynchronously
    /// </summary>
    public static ICommunicationHandle SynchronizeAsync(
        IAsyncCommunicationBackend backend,
        ProcessGroup? group = null)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        return backend.BarrierAsync();
    }
}
