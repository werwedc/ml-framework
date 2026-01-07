using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.HAL.CUDA;

namespace MLFramework.Pipeline
{
    /// <summary>
    /// Manages asynchronous execution for pipeline stages
    /// Overlaps computation and communication to maximize throughput
    /// </summary>
    public class AsyncPipelineExecutor : IDisposable
    {
        private readonly PipelineStage _stage;
        private readonly IPipelineCommunicator _communicator;
        private readonly StreamManager _computeStreamManager;
        private readonly StreamManager _commStreamManager;
        private readonly List<AsyncOperation> _activeOperations;
        private readonly int _disposed;
        private readonly int _numComputeStreams;
        private readonly int _numCommStreams;

        /// <summary>
        /// Number of CUDA streams for compute
        /// </summary>
        public int NumComputeStreams => _numComputeStreams;

        /// <summary>
        /// Number of CUDA streams for communication
        /// </summary>
        public int NumCommStreams => _numCommStreams;

        /// <summary>
        /// Count of currently active operations
        /// </summary>
        public int ActiveOperationsCount => _activeOperations.Count;

        public AsyncPipelineExecutor(
            PipelineStage stage,
            IPipelineCommunicator communicator,
            int numStreams = 2)
        {
            _stage = stage ?? throw new ArgumentNullException(nameof(stage));
            _communicator = communicator ?? throw new ArgumentNullException(nameof(communicator));

            if (numStreams <= 0)
            {
                throw new ArgumentException("Number of streams must be greater than 0", nameof(numStreams));
            }

            _numComputeStreams = numStreams;
            _numCommStreams = numStreams;

            // Get the device from the stage
            var device = _stage.Device as CudaDevice;
            if (device == null)
            {
                throw new ArgumentException("Stage must be on a CUDA device", nameof(stage));
            }

            // Create stream managers
            _computeStreamManager = new StreamManager(device, _numComputeStreams);
            _commStreamManager = new StreamManager(device, _numCommStreams);
            _activeOperations = new List<AsyncOperation>();
            _disposed = 0;
        }

        /// <summary>
        /// Execute forward pass asynchronously on a specific stream
        /// </summary>
        /// <param name="input">Input tensor</param>
        /// <param name="streamIndex">Stream index</param>
        /// <returns>Task that completes when forward pass is done</returns>
        public async Task<Tensor> ForwardAsync(Tensor input, int streamIndex = 0)
        {
            ThrowIfDisposed();

            if (input == null)
                throw new ArgumentNullException(nameof(input));

            var stream = GetComputeStream(streamIndex);

            // Create task for forward pass
            var task = Task.Run(() =>
            {
                try
                {
                    // In a real implementation, we would set the stream context
                    // before executing the forward pass
                    return _stage.Forward(input);
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Forward pass failed on stream {streamIndex}", ex);
                }
            });

            // Track the operation
            var operation = new AsyncOperation(
                Guid.NewGuid(),
                OperationType.Forward,
                -1, // micro-batch index not tracked at this level
                streamIndex,
                task);

            lock (_activeOperations)
            {
                _activeOperations.Add(operation);
            }

            try
            {
                await task;
                return task.Result;
            }
            finally
            {
                lock (_activeOperations)
                {
                    _activeOperations.Remove(operation);
                }
            }
        }

        /// <summary>
        /// Execute backward pass asynchronously on a specific stream
        /// </summary>
        /// <param name="gradOutput">Gradient tensor</param>
        /// <param name="streamIndex">Stream index</param>
        /// <returns>Task that completes when backward pass is done</returns>
        public async Task<Tensor> BackwardAsync(Tensor gradOutput, int streamIndex = 0)
        {
            ThrowIfDisposed();

            if (gradOutput == null)
                throw new ArgumentNullException(nameof(gradOutput));

            var stream = GetComputeStream(streamIndex);

            // Create task for backward pass
            var task = Task.Run(() =>
            {
                try
                {
                    // In a real implementation, we would use autograd to compute gradients
                    // For now, we return the gradient as-is
                    return gradOutput;
                }
                catch (Exception ex)
                {
                    throw new InvalidOperationException($"Backward pass failed on stream {streamIndex}", ex);
                }
            });

            // Track the operation
            var operation = new AsyncOperation(
                Guid.NewGuid(),
                OperationType.Backward,
                -1, // micro-batch index not tracked at this level
                streamIndex,
                task);

            lock (_activeOperations)
            {
                _activeOperations.Add(operation);
            }

            try
            {
                await task;
                return task.Result;
            }
            finally
            {
                lock (_activeOperations)
                {
                    _activeOperations.Remove(operation);
                }
            }
        }

        /// <summary>
        /// Send tensor asynchronously on communication stream
        /// </summary>
        public async Task<Tensor> SendAsync(Tensor tensor, int destinationRank, int streamIndex = 0)
        {
            ThrowIfDisposed();

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            var stream = GetCommStream(streamIndex);

            // Create task for send operation
            var task = _communicator.SendAsync(tensor, destinationRank);

            // Track the operation
            var operation = new AsyncOperation(
                Guid.NewGuid(),
                OperationType.SendForward,
                -1, // micro-batch index not tracked at this level
                streamIndex,
                task);

            lock (_activeOperations)
            {
                _activeOperations.Add(operation);
            }

            try
            {
                await task;
                return tensor;
            }
            finally
            {
                lock (_activeOperations)
                {
                    _activeOperations.Remove(operation);
                }
            }
        }

        /// <summary>
        /// Receive tensor asynchronously on communication stream
        /// </summary>
        public async Task<Tensor> ReceiveAsync(int sourceRank, int streamIndex = 0)
        {
            ThrowIfDisposed();

            var stream = GetCommStream(streamIndex);

            // Create task for receive operation
            var task = _communicator.ReceiveAsync(sourceRank);

            // Track the operation
            var operation = new AsyncOperation(
                Guid.NewGuid(),
                OperationType.ReceiveForward,
                -1, // micro-batch index not tracked at this level
                streamIndex,
                task);

            lock (_activeOperations)
            {
                _activeOperations.Add(operation);
            }

            try
            {
                var result = await task;
                return result;
            }
            finally
            {
                lock (_activeOperations)
                {
                    _activeOperations.Remove(operation);
                }
            }
        }

        /// <summary>
        /// Synchronize all compute streams
        /// </summary>
        public Task SyncComputeAsync()
        {
            ThrowIfDisposed();
            return _computeStreamManager.SynchronizeAllAsync();
        }

        /// <summary>
        /// Synchronize all communication streams
        /// </summary>
        public Task SyncCommAsync()
        {
            ThrowIfDisposed();
            return _commStreamManager.SynchronizeAllAsync();
        }

        /// <summary>
        /// Synchronize all streams
        /// </summary>
        public async Task SyncAllAsync()
        {
            ThrowIfDisposed();

            // Wait for all active operations to complete
            List<AsyncOperation> operationsCopy;
            lock (_activeOperations)
            {
                operationsCopy = new List<AsyncOperation>(_activeOperations);
            }

            await Task.WhenAll(operationsCopy.ConvertAll(op => op.Task));

            // Synchronize all streams
            await Task.WhenAll(
                _computeStreamManager.SynchronizeAllAsync(),
                _commStreamManager.SynchronizeAllAsync());
        }

        /// <summary>
        /// Get stream for a specific micro-batch (compute stream)
        /// </summary>
        public CudaStream GetComputeStream(int microBatchIndex)
        {
            ThrowIfDisposed();

            // Use round-robin to select stream based on micro-batch index
            int streamIndex = microBatchIndex % _numComputeStreams;
            return _computeStreamManager.GetStream(streamIndex);
        }

        /// <summary>
        /// Get communication stream for a specific micro-batch
        /// </summary>
        public CudaStream GetCommStream(int microBatchIndex)
        {
            ThrowIfDisposed();

            // Use round-robin to select stream based on micro-batch index
            int streamIndex = microBatchIndex % _numCommStreams;
            return _commStreamManager.GetStream(streamIndex);
        }

        /// <summary>
        /// Get compute stream by index
        /// </summary>
        public CudaStream GetComputeStreamByIndex(int index)
        {
            ThrowIfDisposed();
            return _computeStreamManager.GetStream(index);
        }

        /// <summary>
        /// Get communication stream by index
        /// </summary>
        public CudaStream GetCommStreamByIndex(int index)
        {
            ThrowIfDisposed();
            return _commStreamManager.GetStream(index);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed == 1)
                throw new ObjectDisposedException(nameof(AsyncPipelineExecutor));
        }

        public void Dispose()
        {
            if (_disposed == 1)
                return;

            _computeStreamManager.Dispose();
            _commStreamManager.Dispose();
        }
    }
}
