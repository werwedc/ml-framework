namespace MLFramework.Communication.Tests;

using MLFramework.Communication;
using RitterFramework.Core.Tensor;
using System;
using System.Threading.Tasks;

/// <summary>
/// Mock implementation of ICommunicationBackend for testing
/// </summary>
public class MockCommunicationBackend : ICommunicationBackend
{
    public int Rank { get; }
    public int WorldSize { get; }
    public string BackendName => "MockBackend";
    private bool _disposed;

    public MockCommunicationBackend(int rank, int worldSize)
    {
        Rank = rank;
        WorldSize = worldSize;
        _disposed = false;
    }

    public void Broadcast(Tensor tensor, int rootRank)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));
        
        // Mock implementation - in real scenario, this would broadcast data
        // For testing, we just verify parameters are valid
    }

    public Tensor Reduce(Tensor tensor, ReduceOp operation, int rootRank)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // Mock implementation - return a copy of the tensor
        // In real scenario, this would reduce data from all ranks
        var resultData = new float[tensor.Size];
        Array.Copy(tensor.Data, resultData, tensor.Size);
        return new Tensor(resultData, tensor.Shape, tensor.RequiresGrad, tensor.Dtype);
    }

    public Tensor AllReduce(Tensor tensor, ReduceOp operation)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // Mock implementation - return a copy of the tensor
        var resultData = new float[tensor.Size];
        Array.Copy(tensor.Data, resultData, tensor.Size);
        return new Tensor(resultData, tensor.Shape, tensor.RequiresGrad, tensor.Dtype);
    }

    public Tensor AllGather(Tensor tensor)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // Mock implementation - return a copy of the tensor
        var resultData = new float[tensor.Size];
        Array.Copy(tensor.Data, resultData, tensor.Size);
        return new Tensor(resultData, tensor.Shape, tensor.RequiresGrad, tensor.Dtype);
    }

    public Tensor ReduceScatter(Tensor tensor, ReduceOp operation)
    {
        if (tensor == null)
            throw new ArgumentNullException(nameof(tensor));

        // Mock implementation - return a copy of the tensor
        var resultData = new float[tensor.Size];
        Array.Copy(tensor.Data, resultData, tensor.Size);
        return new Tensor(resultData, tensor.Shape, tensor.RequiresGrad, tensor.Dtype);
    }

    public void Barrier()
    {
        // Mock implementation - no-op
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
        }
    }
}

/// <summary>
/// Mock implementation of IAsyncCommunicationBackend for testing
/// </summary>
public class MockAsyncCommunicationBackend : MockCommunicationBackend, IAsyncCommunicationBackend
{
    public MockAsyncCommunicationBackend(int rank, int worldSize) 
        : base(rank, worldSize)
    {
    }

    public ICommunicationHandle BroadcastAsync(Tensor tensor, int rootRank)
    {
        // Simulate async operation
        var task = Task.Run(() =>
        {
            Broadcast(tensor, rootRank);
            return tensor;
        });
        return new Operations.PendingOperationHandle(task);
    }

    public ICommunicationHandle AllReduceAsync(Tensor tensor, ReduceOp operation)
    {
        // Simulate async operation
        var task = Task.Run(() =>
        {
            return AllReduce(tensor, operation);
        });
        return new Operations.PendingOperationHandle(task);
    }

    public ICommunicationHandle BarrierAsync()
    {
        // Simulate async operation
        var task = Task.Run(() =>
        {
            Barrier();
            return Tensor.FromArray(new float[] { });
        });
        return new Operations.PendingOperationHandle(task);
    }

    // AllGatherAsync and ReduceScatterAsync are not part of IAsyncCommunicationBackend interface
    // but can be added for testing purposes if needed
    public ICommunicationHandle AllGatherAsync(Tensor tensor)
    {
        var task = Task.Run(() => AllGather(tensor));
        return new Operations.PendingOperationHandle(task);
    }

    public ICommunicationHandle ReduceScatterAsync(Tensor tensor, ReduceOp operation)
    {
        var task = Task.Run(() => ReduceScatter(tensor, operation));
        return new Operations.PendingOperationHandle(task);
    }
}
