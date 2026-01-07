using MLFramework.Distributed;
using RitterFramework.Core.Tensor;
using System;
using System.Threading.Tasks;

namespace MLFramework.Layers.TensorParallel
{
    /// <summary>
    /// Utility class for Tensor Parallel operations.
    /// Provides static methods for accessing TP context and communication primitives.
    /// </summary>
    public static class TensorParallel
    {
        private static IProcessGroup? _defaultProcessGroup;

        /// <summary>
        /// Gets or sets the default process group for TP operations.
        /// </summary>
        public static IProcessGroup DefaultProcessGroup
        {
            get => _defaultProcessGroup ?? throw new InvalidOperationException(
                "Default process group not initialized. Call Initialize() first.");
            set => _defaultProcessGroup = value;
        }

        /// <summary>
        /// Gets the world size (total number of processes) in the TP group.
        /// </summary>
        public static int GetWorldSize()
        {
            return DefaultProcessGroup.WorldSize;
        }

        /// <summary>
        /// Gets the rank of this process in the TP group.
        /// </summary>
        public static int GetRank()
        {
            return DefaultProcessGroup.Rank;
        }

        /// <summary>
        /// Gets the communicator (process group) for TP operations.
        /// </summary>
        public static IProcessGroup GetCommunicator()
        {
            return DefaultProcessGroup;
        }

        /// <summary>
        /// Initializes the TP context with the specified process group.
        /// </summary>
        public static void Initialize(IProcessGroup processGroup)
        {
            _defaultProcessGroup = processGroup ?? throw new ArgumentNullException(nameof(processGroup));
        }

        /// <summary>
        /// Checks if TP has been initialized.
        /// </summary>
        public static bool IsInitialized()
        {
            return _defaultProcessGroup != null;
        }
    }

    /// <summary>
    /// Represents a tensor parallel process group.
    /// Wraps the communication primitives needed for TP operations.
    /// </summary>
    public class TensorParallelGroup : IDisposable
    {
        private readonly IProcessGroup _processGroup;
        private bool _disposed;

        /// <summary>
        /// Gets the underlying process group.
        /// </summary>
        public IProcessGroup Communicator => _processGroup;

        /// <summary>
        /// Gets the world size of the TP group.
        /// </summary>
        public int WorldSize => _processGroup.WorldSize;

        /// <summary>
        /// Gets the rank of this process.
        /// </summary>
        public int Rank => _processGroup.Rank;

        /// <summary>
        /// Creates a new TensorParallelGroup.
        /// </summary>
        public TensorParallelGroup(IProcessGroup processGroup)
        {
            _processGroup = processGroup ?? throw new ArgumentNullException(nameof(processGroup));
        }

        /// <summary>
        /// Performs an all-reduce operation synchronously.
        /// </summary>
        public void AllReduce(Tensor tensor, ReduceOp op = ReduceOp.Sum)
        {
            _processGroup.AllReduce(tensor, op);
        }

        /// <summary>
        /// Performs an all-reduce operation asynchronously.
        /// </summary>
        public Task AllReduceAsync(Tensor tensor, ReduceOp op = ReduceOp.Sum)
        {
            return _processGroup.AllReduceAsync(tensor, op);
        }

        /// <summary>
        /// Performs an all-gather operation synchronously.
        /// Gathers data from all ranks and concatenates along the specified dimension.
        /// </summary>
        public void AllGather(Tensor tensor, int dim = -1)
        {
            var task = AllGatherAsync(tensor, dim);
            task.Wait();
        }

        /// <summary>
        /// Performs an all-gather operation asynchronously.
        /// Gathers data from all ranks and concatenates along the specified dimension.
        /// </summary>
        public async Task<Tensor> AllGatherAsync(Tensor tensor, int dim = -1)
        {
            // For a mock implementation, we'll just return the tensor itself
            // In a real implementation with NCCL/Gloo, this would:
            // 1. Gather tensors from all ranks
            // 2. Concatenate them along the specified dimension
            // 3. Return the concatenated tensor

            // Handle negative dimension indexing
            if (dim < 0)
            {
                dim = tensor.Dimensions + dim;
            }

            if (dim < 0 || dim >= tensor.Dimensions)
            {
                throw new ArgumentOutOfRangeException(nameof(dim), $"Dimension {dim} is out of bounds for tensor with {tensor.Dimensions} dimensions");
            }

            // For mock/single-process case, just return the tensor
            if (WorldSize == 1)
            {
                return tensor;
            }

            // In a real distributed implementation, we would:
            // 1. Broadcast the tensor size from rank 0
            // 2. Allocate buffer for gathered tensors
            // 3. Gather from all ranks
            // 4. Concatenate along the specified dimension

            // For now, simulate the operation by replicating the tensor
            int outputDimSize = tensor.Shape[dim] * WorldSize;
            var newShape = (int[])tensor.Shape.Clone();
            newShape[dim] = outputDimSize;

            // Create output tensor
            var outputData = new float[tensor.Size * WorldSize];
            Array.Copy(tensor.Data, 0, outputData, 0, tensor.Size);

            // In a real implementation, we'd copy data from each rank here
            return new Tensor(outputData, newShape, tensor.RequiresGrad, tensor.Dtype);
        }

        /// <summary>
        /// Performs a barrier to synchronize all ranks.
        /// </summary>
        public void Barrier()
        {
            _processGroup.Barrier();
        }

        /// <summary>
        /// Performs a barrier asynchronously to synchronize all ranks.
        /// </summary>
        public Task BarrierAsync()
        {
            return _processGroup.BarrierAsync();
        }

        /// <summary>
        /// Disposes the process group.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _processGroup.Dispose();
                _disposed = true;
            }
        }
    }
}
