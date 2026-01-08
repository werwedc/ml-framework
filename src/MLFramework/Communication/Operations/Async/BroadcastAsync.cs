namespace MLFramework.Communication.Operations.Async;

using MLFramework.Communication;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using System;

/// <summary>
/// Asynchronous broadcast operation
/// </summary>
public static class BroadcastAsync
{
    /// <summary>
    /// Broadcast tensor asynchronously from root rank to all ranks
    /// </summary>
    /// <param name="backend">Async communication backend</param>
    /// <param name="tensor">Tensor to broadcast</param>
    /// <param name="rootRank">Rank that will broadcast the data</param>
    /// <param name="group">Process group (default: world)</param>
    /// <returns>Handle to the ongoing broadcast operation</returns>
    public static ICommunicationHandle BroadcastTensorAsync(
        IAsyncCommunicationBackend backend,
        Tensor tensor,
        int rootRank,
        ProcessGroup? group = null)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (rootRank < 0 || rootRank >= backend.WorldSize)
            throw new ArgumentOutOfRangeException(nameof(rootRank));

        if (group != null)
        {
            // Check if rank is in the group
            // Note: For the existing ProcessGroup, we'll just do basic validation
        }

        return backend.BroadcastAsync(tensor, rootRank);
    }
}
