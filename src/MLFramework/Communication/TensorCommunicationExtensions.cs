namespace MLFramework.Communication;

using RitterFramework.Core.Tensor;

/// <summary>
/// Extension methods for Tensor communication
/// </summary>
public static class TensorCommunicationExtensions
{
    /// <summary>
    /// Broadcast this tensor from root rank to all ranks
    /// </summary>
    public static void BroadcastToAll(this Tensor tensor, ICommunicationBackend backend, int rootRank)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        backend.Broadcast(tensor, rootRank);
    }

    /// <summary>
    /// Perform all-reduce on this tensor in place
    /// </summary>
    public static Tensor AllReduceInPlace(this Tensor tensor, ICommunicationBackend backend, ReduceOp operation)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        return backend.AllReduce(tensor, operation);
    }

    /// <summary>
    /// Scatter tensor across ranks (each rank gets a chunk)
    /// </summary>
    public static Tensor Scatter(this Tensor tensor, ICommunicationBackend backend, int rank)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        // Implementation using AllGather + indexing
        var gathered = backend.AllGather(tensor);
        return SliceForRank(gathered, rank, backend.WorldSize);
    }

    /// <summary>
    /// Slice tensor for a specific rank
    /// </summary>
    private static Tensor SliceForRank(Tensor tensor, int rank, int worldSize)
    {
        // Calculate chunk size and return appropriate slice
        // This is a placeholder - actual implementation depends on tensor layout
        throw new NotImplementedException("SliceForRank needs to be implemented based on tensor layout");
    }
}
