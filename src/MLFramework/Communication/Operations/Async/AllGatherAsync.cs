namespace MLFramework.Communication.Operations.Async;

using MLFramework.Communication;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using System.Threading.Tasks;

/// <summary>
/// Asynchronous all-gather operation
/// </summary>
public static class AllGatherAsync
{
    /// <summary>
    /// All-gather tensor asynchronously
    /// </summary>
    public static ICommunicationHandle AllGatherTensorAsync(
        IAsyncCommunicationBackend backend,
        Tensor tensor,
        ProcessGroup? group = null)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // Create async task
        var task = Task.Run(() => backend.AllGather(tensor));
        return new PendingOperationHandle(task);
    }

    /// <summary>
    /// Gather list of tensors asynchronously
    /// </summary>
    public static ICommunicationHandle AllGatherTensorsAsync(
        IAsyncCommunicationBackend backend,
        Tensor tensor,
        ProcessGroup? group = null)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // For this version, we just return the gathered tensor directly
        // Use AllGatherTensorAsync instead and split manually if needed
        return AllGatherTensorAsync(backend, tensor, group);
    }
}
