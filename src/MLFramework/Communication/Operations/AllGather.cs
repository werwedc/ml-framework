namespace MLFramework.Communication.Operations;

using MLFramework.Communication;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;
using System.Collections.Generic;

/// <summary>
/// Synchronous all-gather operation
/// </summary>
public static class AllGather
{
    /// <summary>
    /// Gather tensor data from all ranks and distribute to all ranks
    /// </summary>
    /// <param name="backend">Communication backend</param>
    /// <param name="tensor">Tensor to gather</param>
    /// <param name="group">Process group (default: world)</param>
    /// <returns>Gathered tensor concatenated from all ranks</returns>
    public static Tensor AllGatherTensor(
        ICommunicationBackend backend,
        Tensor tensor,
        ProcessGroup? group = null)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        return backend.AllGather(tensor);
    }

    /// <summary>
    /// Gather list of tensors from all ranks
    /// </summary>
    /// <returns>List of tensors gathered from each rank</returns>
    public static List<Tensor> AllGatherTensors(
        ICommunicationBackend backend,
        Tensor tensor,
        ProcessGroup? group = null)
    {
        if (backend == null)
            throw new ArgumentNullException(nameof(backend));

        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // Gather all data
        var gathered = backend.AllGather(tensor);

        // Split into per-rank tensors
        return SplitGatheredTensor(gathered, backend.WorldSize);
    }

    /// <summary>
    /// Split gathered tensor into per-rank chunks
    /// </summary>
    internal static List<Tensor> SplitGatheredTensor(Tensor gathered, int worldSize)
    {
        // Calculate chunk size
        long totalElements = gathered.Size;
        long chunkSize = totalElements / worldSize;

        // Split into worldSize chunks
        var result = new List<Tensor>();
        for (int i = 0; i < worldSize; i++)
        {
            var startIdx = i * chunkSize;
            var endIdx = (i == worldSize - 1) ? totalElements : (i + 1) * chunkSize;
            var chunk = SliceTensor(gathered, startIdx, (int)(endIdx - startIdx));
            result.Add(chunk);
        }

        return result;
    }

    /// <summary>
    /// Create a slice of a tensor from the given index
    /// </summary>
    private static Tensor SliceTensor(Tensor tensor, long startIndex, int length)
    {
        var sliceData = new float[length];
        for (int i = 0; i < length; i++)
        {
            sliceData[i] = tensor.Data[startIndex + i];
        }
        return new Tensor(sliceData, new int[] { length }, tensor.RequiresGrad, tensor.Dtype);
    }
}
