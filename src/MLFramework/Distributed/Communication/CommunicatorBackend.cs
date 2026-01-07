namespace MLFramework.Distributed.Communication;

using RitterFramework.Core.Tensor;
using System.Threading.Tasks;

/// <summary>
/// Abstract base class for communication backends.
/// Provides common functionality for all communicator implementations.
/// </summary>
public abstract class CommunicatorBackend : ICommunicator
{
    protected readonly int _worldSize;
    protected readonly int _rank;

    protected CommunicatorBackend(int worldSize, int rank)
    {
        if (worldSize <= 0)
        {
            throw new ArgumentException("World size must be positive", nameof(worldSize));
        }

        if (rank < 0 || rank >= worldSize)
        {
            throw new ArgumentException($"Rank must be in range [0, {worldSize - 1}]", nameof(rank));
        }

        _worldSize = worldSize;
        _rank = rank;
    }

    public int WorldSize => _worldSize;
    public int Rank => _rank;

    public abstract Task<Tensor> AllReduceAsync(Tensor tensor, ReduceOperation operation);
    public abstract Task<Tensor> AllGatherAsync(Tensor tensor, int dim);
    public abstract Task<Tensor> ReduceScatterAsync(Tensor tensor, ReduceOperation operation);
    public abstract Task<Tensor> BroadcastAsync(Tensor tensor, int root);
    public abstract Task BarrierAsync();

    public abstract void Dispose();
}
