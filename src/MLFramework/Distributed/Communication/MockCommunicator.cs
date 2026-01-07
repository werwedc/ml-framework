namespace MLFramework.Distributed.Communication;

using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

/// <summary>
/// Mock communicator for testing without actual distributed hardware.
/// Simulates collective operations using in-memory shared state.
/// </summary>
public class MockCommunicator : CommunicatorBackend
{
    private readonly Dictionary<int, List<Tensor>> _sharedMemory;
    private readonly object _lock;
    private bool _disposed;

    /// <summary>
    /// Creates a new mock communicator.
    /// </summary>
    /// <param name="worldSize">Total number of processes</param>
    /// <param name="rank">Rank of this process</param>
    public MockCommunicator(int worldSize, int rank)
        : base(worldSize, rank)
    {
        _sharedMemory = new Dictionary<int, List<Tensor>>();
        _lock = new object();
        _disposed = false;

        for (int i = 0; i < worldSize; i++)
        {
            _sharedMemory[i] = new List<Tensor>();
        }
    }

    public override async Task<Tensor> AllReduceAsync(Tensor tensor, ReduceOperation operation)
    {
        // Store this rank's tensor
        lock (_lock)
        {
            _sharedMemory[_rank].Add(tensor.Clone());
        }

        // Wait for all ranks to contribute (simulated delay)
        await Task.Delay(1);

        // Collect all tensors
        Tensor[] allTensors;
        lock (_lock)
        {
            allTensors = _sharedMemory.Values
                .Where(list => list.Count > _rank)
                .Select(list => list[_rank])
                .ToArray();

            // Clear shared memory for this operation index
            if (allTensors.Length == _worldSize)
            {
                foreach (var list in _sharedMemory.Values)
                {
                    if (list.Count > _rank)
                    {
                        list.RemoveAt(0);
                    }
                }
            }
        }

        var result = tensor.Clone();

        // Apply reduction operation
        for (int i = 1; i < allTensors.Length; i++)
        {
            switch (operation)
            {
                case ReduceOperation.Sum:
                    result = AddTensors(result, allTensors[i]);
                    break;
                case ReduceOperation.Max:
                    result = MaxTensors(result, allTensors[i]);
                    break;
                case ReduceOperation.Min:
                    result = MinTensors(result, allTensors[i]);
                    break;
                case ReduceOperation.Product:
                    result = MultiplyTensors(result, allTensors[i]);
                    break;
                case ReduceOperation.Avg:
                    result = AddTensors(result, allTensors[i]);
                    break;
            }
        }

        // If operation is Avg, divide by count
        if (operation == ReduceOperation.Avg && allTensors.Length > 0)
        {
            result = DivideByScalar(result, allTensors.Length);
        }

        return result;
    }

    public override async Task<Tensor> AllGatherAsync(Tensor tensor, int dim = 0)
    {
        // Handle negative dimension indexing
        if (dim < 0)
        {
            dim = tensor.Dimensions + dim;
        }

        if (dim < 0 || dim >= tensor.Dimensions)
        {
            throw new ArgumentOutOfRangeException(nameof(dim),
                $"Dimension {dim} is out of bounds for tensor with {tensor.Dimensions} dimensions");
        }

        // Store this rank's tensor
        lock (_lock)
        {
            _sharedMemory[_rank].Add(tensor.Clone());
        }

        // Wait for all ranks to contribute
        await Task.Delay(1);

        // Collect all tensors
        Tensor[] allTensors;
        lock (_lock)
        {
            allTensors = _sharedMemory.Values
                .Where(list => list.Count > _rank)
                .Select(list => list[_rank])
                .ToArray();

            // Clear shared memory
            if (allTensors.Length == _worldSize)
            {
                foreach (var list in _sharedMemory.Values)
                {
                    if (list.Count > _rank)
                    {
                        list.RemoveAt(0);
                    }
                }
            }
        }

        // For mock/single-process case, just return the tensor
        if (_worldSize == 1)
        {
            return tensor;
        }

        // Concatenate along the specified dimension
        return ConcatenateTensors(allTensors, dim);
    }

    public override async Task<Tensor> ReduceScatterAsync(Tensor tensor, ReduceOperation operation)
    {
        // First reduce all tensors
        var reduced = await AllReduceAsync(tensor, operation);

        // Then scatter: each rank gets a slice (last dimension)
        int dimSize = reduced.Shape[^1];
        int chunkSize = dimSize / _worldSize;

        if (dimSize % _worldSize != 0)
        {
            throw new InvalidOperationException(
                $"Last dimension ({dimSize}) must be divisible by world size ({_worldSize})");
        }

        int startIdx = _rank * chunkSize;
        int endIdx = startIdx + chunkSize;

        return reduced.Slice(-1, startIdx, endIdx);
    }

    public override async Task<Tensor> BroadcastAsync(Tensor tensor, int root)
    {
        if (root < 0 || root >= _worldSize)
        {
            throw new ArgumentOutOfRangeException(nameof(root),
                $"Root rank must be in range [0, {_worldSize - 1}]");
        }

        if (_rank == root)
        {
            // Root stores its tensor
            lock (_lock)
            {
                _sharedMemory[root].Add(tensor.Clone());
            }
        }

        await Task.Delay(1);

        // Other ranks get tensor from root
        if (_rank != root)
        {
            Tensor? rootTensor;
            lock (_lock)
            {
                rootTensor = _sharedMemory[root].Count > 0
                    ? _sharedMemory[root][0]
                    : null;
            }

            if (rootTensor != null)
            {
                lock (_lock)
                {
                    if (_sharedMemory[root].Count > 0)
                    {
                        _sharedMemory[root].Clear();
                    }
                }
                return rootTensor.Clone();
            }
        }

        return tensor.Clone();
    }

    public override Task BarrierAsync()
    {
        // In mock implementation, just complete immediately
        return Task.CompletedTask;
    }

    public override void Dispose()
    {
        if (!_disposed)
        {
            lock (_lock)
            {
                _sharedMemory.Clear();
            }
            _disposed = true;
        }
    }

    #region Helper Methods

    private static Tensor AddTensors(Tensor a, Tensor b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
        {
            throw new ArgumentException("Tensor shapes must match for addition");
        }

        var resultData = new float[a.Size];
        for (int i = 0; i < a.Size; i++)
        {
            resultData[i] = a.Data[i] + b.Data[i];
        }

        return new Tensor(resultData, a.Shape, a.RequiresGrad || b.RequiresGrad, a.Dtype);
    }

    private static Tensor MaxTensors(Tensor a, Tensor b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
        {
            throw new ArgumentException("Tensor shapes must match for max operation");
        }

        var resultData = new float[a.Size];
        for (int i = 0; i < a.Size; i++)
        {
            resultData[i] = Math.Max(a.Data[i], b.Data[i]);
        }

        return new Tensor(resultData, a.Shape, a.RequiresGrad || b.RequiresGrad, a.Dtype);
    }

    private static Tensor MinTensors(Tensor a, Tensor b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
        {
            throw new ArgumentException("Tensor shapes must match for min operation");
        }

        var resultData = new float[a.Size];
        for (int i = 0; i < a.Size; i++)
        {
            resultData[i] = Math.Min(a.Data[i], b.Data[i]);
        }

        return new Tensor(resultData, a.Shape, a.RequiresGrad || b.RequiresGrad, a.Dtype);
    }

    private static Tensor MultiplyTensors(Tensor a, Tensor b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
        {
            throw new ArgumentException("Tensor shapes must match for multiplication");
        }

        var resultData = new float[a.Size];
        for (int i = 0; i < a.Size; i++)
        {
            resultData[i] = a.Data[i] * b.Data[i];
        }

        return new Tensor(resultData, a.Shape, a.RequiresGrad || b.RequiresGrad, a.Dtype);
    }

    private static Tensor DivideByScalar(Tensor tensor, int divisor)
    {
        var resultData = new float[tensor.Size];
        for (int i = 0; i < tensor.Size; i++)
        {
            resultData[i] = tensor.Data[i] / divisor;
        }

        return new Tensor(resultData, tensor.Shape, tensor.RequiresGrad, tensor.Dtype);
    }

    private static Tensor ConcatenateTensors(Tensor[] tensors, int dim)
    {
        if (tensors.Length == 0)
        {
            throw new ArgumentException("Cannot concatenate empty array of tensors");
        }

        var firstShape = tensors[0].Shape;
        int outputDimSize = 0;

        // Validate all tensors have the same shape except for the concatenation dimension
        foreach (var tensor in tensors)
        {
            if (tensor.Dimensions != firstShape.Length)
            {
                throw new ArgumentException("All tensors must have the same number of dimensions");
            }

            for (int d = 0; d < tensor.Dimensions; d++)
            {
                if (d != dim && tensor.Shape[d] != firstShape[d])
                {
                    throw new ArgumentException(
                        $"All tensors must have the same size for dimension {d}");
                }
            }

            outputDimSize += tensor.Shape[dim];
        }

        // Calculate output shape
        var outputShape = (int[])firstShape.Clone();
        outputShape[dim] = outputDimSize;

        // Calculate output size
        int outputSize = 1;
        foreach (var dimSize in outputShape)
        {
            outputSize *= dimSize;
        }

        var outputData = new float[outputSize];
        int outputOffset = 0;

        // Copy data from each tensor
        foreach (var tensor in tensors)
        {
            // Calculate stride before concatenation dimension
            int strideBefore = 1;
            for (int i = 0; i < dim; i++)
            {
                strideBefore *= tensor.Shape[i];
            }

            // Calculate stride after concatenation dimension
            int strideAfter = 1;
            for (int i = dim + 1; i < tensor.Dimensions; i++)
            {
                strideAfter *= tensor.Shape[i];
            }

            int dimSize = tensor.Shape[dim];

            for (int before = 0; before < strideBefore; before++)
            {
                int inputBase = before * dimSize * strideAfter;
                int outputBase = outputOffset + before * outputDimSize * strideAfter;

                for (int d = 0; d < dimSize; d++)
                {
                    for (int after = 0; after < strideAfter; after++)
                    {
                        outputData[outputBase + d * strideAfter + after] =
                            tensor.Data[inputBase + d * strideAfter + after];
                    }
                }
            }

            outputOffset += tensor.Shape[dim] * strideAfter;
        }

        return new Tensor(outputData, outputShape, tensors[0].RequiresGrad, tensors[0].Dtype);
    }

    #endregion
}
