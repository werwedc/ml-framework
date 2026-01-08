namespace MLFramework.Communication.Operations.Async;

using MLFramework.Communication;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;

/// <summary>
/// Asynchronous reduce-scatter operation
/// </summary>
public static class ReduceScatterAsync
{
    /// <summary>
    /// Reduce-scatter tensor asynchronously
    /// </summary>
    public static ICommunicationHandle ReduceScatterTensorAsync(
        IAsyncCommunicationBackend backend,
        Tensor tensor,
        ReduceOp operation,
        ProcessGroup? group = null)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        Operations.Reduce.ValidateReduceOperation(operation, tensor.Dtype);

        var task = System.Threading.Tasks.Task.Run(() => backend.ReduceScatter(tensor, operation));
        return new PendingOperationHandle(task);
    }
}
