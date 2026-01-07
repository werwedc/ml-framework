namespace MLFramework.Communication;

using RitterFramework.Core.Tensor;

/// <summary>
/// Interface for asynchronous communication operations
/// </summary>
public interface IAsyncCommunicationBackend : ICommunicationBackend
{
    /// <summary>
    /// Non-blocking broadcast operation
    /// </summary>
    ICommunicationHandle BroadcastAsync(Tensor tensor, int rootRank);

    /// <summary>
    /// Non-blocking all-reduce operation
    /// </summary>
    ICommunicationHandle AllReduceAsync(Tensor tensor, ReduceOp operation);

    /// <summary>
    /// Non-blocking barrier operation
    /// </summary>
    ICommunicationHandle BarrierAsync();
}
