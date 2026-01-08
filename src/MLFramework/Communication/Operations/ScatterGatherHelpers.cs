namespace MLFramework.Communication.Operations;

using MLFramework.Communication;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;

/// <summary>
/// Helper methods for scatter and gather patterns
/// </summary>
public static class ScatterGatherHelpers
{
    /// <summary>
    /// Scatter tensor across ranks (each rank gets a contiguous chunk)
    /// </summary>
    public static Tensor Scatter(
        ICommunicationBackend backend,
        Tensor tensor,
        ProcessGroup? group = null)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // Use AllGather and then slice for this rank
        var gathered = AllGather.AllGatherTensor(backend, tensor, group);
        return SliceForRank(gathered, backend.Rank, backend.WorldSize);
    }

    /// <summary>
    /// Gather tensor chunks from all ranks (inverse of Scatter)
    /// </summary>
    public static Tensor Gather(
        ICommunicationBackend backend,
        Tensor tensor,
        int rootRank = 0,
        ProcessGroup? group = null)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // AllGather returns data on all ranks
        return AllGather.AllGatherTensor(backend, tensor, group);
    }

    /// <summary>
    /// Slice tensor to get chunk for specific rank
    /// </summary>
    private static Tensor SliceForRank(Tensor tensor, int rank, int worldSize)
    {
        long totalElements = tensor.Size;
        long chunkSize = totalElements / worldSize;

        long startIdx = rank * chunkSize;
        long endIdx = (rank == worldSize - 1) ? totalElements : (rank + 1) * chunkSize;

        var sliceData = new float[endIdx - startIdx];
        for (int i = 0; i < endIdx - startIdx; i++)
        {
            sliceData[i] = tensor.Data[startIdx + i];
        }

        return new Tensor(sliceData, new int[] { (int)(endIdx - startIdx) }, tensor.RequiresGrad, tensor.Dtype);
    }
}
