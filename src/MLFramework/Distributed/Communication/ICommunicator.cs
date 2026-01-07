namespace MLFramework.Distributed.Communication;

using RitterFramework.Core.Tensor;
using System.Threading.Tasks;

/// <summary>
/// Interface for collective communication operations required by Tensor Parallelism.
/// This provides a high-level wrapper for collective operations that hides backend-specific details.
/// </summary>
public interface ICommunicator : IDisposable
{
    /// <summary>
    /// Gets the world size (total number of processes/ranks).
    /// </summary>
    int WorldSize { get; }

    /// <summary>
    /// Gets the rank of this process (0 to WorldSize-1).
    /// </summary>
    int Rank { get; }

    /// <summary>
    /// Performs all-reduce operation: sums tensors across all ranks and distributes result to all.
    /// Returns a new tensor with the reduced result.
    /// </summary>
    Task<Tensor> AllReduceAsync(Tensor tensor, ReduceOperation operation = ReduceOperation.Sum);

    /// <summary>
    /// Performs all-gather operation: gathers tensor shards from all ranks and concatenates.
    /// Returns a new tensor with the gathered result.
    /// </summary>
    Task<Tensor> AllGatherAsync(Tensor tensor, int dim = 0);

    /// <summary>
    /// Performs reduce-scatter operation: reduces then scatters result chunks.
    /// Returns a new tensor with the scattered result.
    /// </summary>
    Task<Tensor> ReduceScatterAsync(Tensor tensor, ReduceOperation operation = ReduceOperation.Sum);

    /// <summary>
    /// Broadcasts a tensor from a specific rank to all other ranks.
    /// Returns a new tensor with the broadcasted result.
    /// </summary>
    Task<Tensor> BroadcastAsync(Tensor tensor, int root);

    /// <summary>
    /// Barrier: synchronizes all ranks.
    /// </summary>
    Task BarrierAsync();
}
