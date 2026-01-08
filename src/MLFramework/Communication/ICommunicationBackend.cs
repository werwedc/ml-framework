namespace MLFramework.Communication;

using RitterFramework.Core.Tensor;

/// <summary>
/// Base interface for all communication backends
/// </summary>
public interface ICommunicationBackend : IDisposable
{
    /// <summary>
    /// Gets the rank of this process in the communication group
    /// </summary>
    int Rank { get; }

    /// <summary>
    /// Gets the total number of processes in the communication group
    /// </summary>
    int WorldSize { get; }

    /// <summary>
    /// Gets the backend name for logging/debugging
    /// </summary>
    string BackendName { get; }

    /// <summary>
    /// Gets the device type for this backend
    /// </summary>
    DeviceType Device { get; }

    /// <summary>
    /// Broadcast tensor data from root rank to all ranks
    /// </summary>
    void Broadcast(Tensor tensor, int rootRank);

    /// <summary>
    /// Reduce tensor data from all ranks to root rank
    /// </summary>
    Tensor Reduce(Tensor tensor, ReduceOp operation, int rootRank);

    /// <summary>
    /// AllReduce: combine data from all ranks and distribute to all
    /// </summary>
    Tensor AllReduce(Tensor tensor, ReduceOp operation);

    /// <summary>
    /// AllGather: combine data from all ranks and distribute full dataset to all
    /// </summary>
    Tensor AllGather(Tensor tensor);

    /// <summary>
    /// ReduceScatter: combine data from all ranks and scatter chunks
    /// </summary>
    Tensor ReduceScatter(Tensor tensor, ReduceOp operation);

    /// <summary>
    /// Barrier: synchronize all ranks
    /// </summary>
    void Barrier();
}
