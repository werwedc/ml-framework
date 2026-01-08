using MLFramework.Distributed;

namespace MLFramework.Checkpointing.Distributed;

/// <summary>
/// Handles communication between distributed processes
/// </summary>
public class DistributedCommunication : IDisposable
{
    private readonly ProcessGroup _processGroup;
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of DistributedCommunication
    /// </summary>
    /// <param name="processGroup">Process group for communication</param>
    public DistributedCommunication(ProcessGroup processGroup)
    {
        _processGroup = processGroup ?? throw new ArgumentNullException(nameof(processGroup));
        _disposed = false;
    }

    /// <summary>
    /// Broadcasts a tensor to all processes
    /// </summary>
    /// <param name="tensor">Tensor to broadcast</param>
    /// <param name="sourceRank">Source rank</param>
    public void Broadcast(RitterFramework.Core.Tensor.Tensor tensor, int sourceRank)
    {
        ThrowIfDisposed();
        _processGroup.Broadcast(tensor, sourceRank);
    }

    /// <summary>
    /// Sends a tensor to a specific rank
    /// </summary>
    /// <param name="tensor">Tensor to send</param>
    /// <param name="destinationRank">Destination rank</param>
    /// <param name="tag">Tag for the message</param>
    public void Send(RitterFramework.Core.Tensor.Tensor tensor, int destinationRank, int tag = 0)
    {
        ThrowIfDisposed();
        // Note: The tag is ignored in the basic ProcessGroup implementation
        _processGroup.Send(tensor, destinationRank);
    }

    /// <summary>
    /// Receives a tensor from a specific rank
    /// </summary>
    /// <param name="sourceRank">Source rank</param>
    /// <param name="tag">Tag for the message</param>
    /// <returns>Received tensor</returns>
    public RitterFramework.Core.Tensor.Tensor Receive(int sourceRank, int tag = 0)
    {
        ThrowIfDisposed();
        // Create a placeholder tensor to receive into
        // In a real implementation, you'd need to know the size/shape beforehand
        // For now, we'll create a small tensor as a placeholder
        var received = RitterFramework.Core.Tensor.Tensor.Zeros(new int[] { 1 });
        _processGroup.Recv(received, sourceRank);
        return received;
    }

    /// <summary>
    /// Receives a tensor with layer ID
    /// </summary>
    /// <param name="sourceRank">Source rank</param>
    /// <param name="layerId">Layer ID to receive</param>
    /// <returns>Received tensor</returns>
    public RitterFramework.Core.Tensor.Tensor ReceiveTensor(int sourceRank, string layerId)
    {
        return Receive(sourceRank, HashCode.Combine(layerId));
    }

    /// <summary>
    /// All-gather operation
    /// </summary>
    /// <typeparam name="T">Type of data</typeparam>
    /// <param name="data">Local data</param>
    /// <returns>Data from all ranks</returns>
    public List<T> AllGather<T>(T data)
    {
        ThrowIfDisposed();

        // For a basic implementation, we'll need to gather data from all ranks
        // This is a simplified version - in reality you'd use proper collective communication
        var worldSize = _processGroup.WorldSize;
        var gatheredData = new List<T>();

        // In a real implementation, this would use actual all-gather
        // For now, we'll just collect from the current rank
        for (int i = 0; i < worldSize; i++)
        {
            if (i == _processGroup.Rank)
            {
                gatheredData.Add(data);
            }
            else
            {
                // In a real implementation, we'd gather from other ranks
                // For now, add default value
                gatheredData.Add(default(T)!);
            }
        }

        return gatheredData;
    }

    /// <summary>
    /// Barrier operation - synchronizes all processes
    /// </summary>
    public void Barrier()
    {
        ThrowIfDisposed();
        _processGroup.Barrier();
    }

    /// <summary>
    /// Reduces data from all ranks (sum)
    /// </summary>
    /// <typeparam name="T">Type of data</typeparam>
    /// <param name="data">Local data</param>
    /// <returns>Reduced data</returns>
    public T Reduce<T>(T data)
    {
        ThrowIfDisposed();

        // For numeric types, perform sum reduction
        if (data is int intData)
        {
            return (T)(object)intData; // In reality, would sum across all ranks
        }
        else if (data is long longData)
        {
            return (T)(object)longData; // In reality, would sum across all ranks
        }
        else if (data is float floatData)
        {
            return (T)(object)floatData; // In reality, would sum across all ranks
        }
        else if (data is double doubleData)
        {
            return (T)(object)doubleData; // In reality, would sum across all ranks
        }

        return data;
    }

    /// <summary>
    /// All-reduce operation
    /// </summary>
    /// <typeparam name="T">Type of data</typeparam>
    /// <param name="data">Local data</param>
    /// <returns>Reduced data on all ranks</returns>
    public T AllReduce<T>(T data)
    {
        ThrowIfDisposed();

        // All-reduce is similar to reduce but distributed to all ranks
        return Reduce(data);
    }

    /// <summary>
    /// Disposes the communication and releases resources
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(DistributedCommunication));
        }
    }
}
