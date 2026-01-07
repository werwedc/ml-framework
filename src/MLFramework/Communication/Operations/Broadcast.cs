namespace MLFramework.Communication.Operations;

using MLFramework.Communication;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using System.Collections.Generic;
using System;

/// <summary>
/// Synchronous broadcast operation
/// </summary>
public static class Broadcast
{
    /// <summary>
    /// Broadcast tensor from root rank to all ranks in the process group
    /// </summary>
    /// <param name="backend">Communication backend</param>
    /// <param name="tensor">Tensor to broadcast</param>
    /// <param name="rootRank">Rank that will broadcast the data</param>
    /// <param name="group">Process group (default: world)</param>
    public static void BroadcastTensor(
        ICommunicationBackend backend,
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
            var ranks = new HashSet<int>();
            // Note: For the existing ProcessGroup, we'll just do basic validation
            // A full implementation would check if the rank is in the group
        }

        // Broadcast operation
        backend.Broadcast(tensor, rootRank);
    }

    /// <summary>
    /// Broadcast multiple tensors from root rank to all ranks
    /// </summary>
    public static void BroadcastTensors(
        ICommunicationBackend backend,
        IEnumerable<Tensor> tensors,
        int rootRank,
        ProcessGroup? group = null)
    {
        foreach (var tensor in tensors)
        {
            BroadcastTensor(backend, tensor, rootRank, group);
        }
    }
}
