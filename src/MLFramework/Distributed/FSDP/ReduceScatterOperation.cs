using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using MLFramework.Distributed;
using System;
using System.Threading.Tasks;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Implements Reduce-Scatter communication operation for FSDP.
    /// Reduces and scatters gradients to owning devices.
    /// </summary>
    public class ReduceScatterOperation : IDisposable
    {
        private readonly IProcessGroup _processGroup;
        private readonly Tensor _shardedBuffer;
        private readonly int _shardIndex;
        private readonly ReduceOp _reduceOp;

        /// <summary>
        /// Initialize a new Reduce-Scatter operation.
        /// </summary>
        /// <param name="processGroup">Process group for communication</param>
        /// <param name="fullShape">Shape of the full tensor before scattering</param>
        /// <param name="dataType">Data type of the tensor</param>
        /// <param name="shardIndex">Index of the shard to receive</param>
        /// <param name="reduceOp">Reduction operation (Sum, Avg, etc.)</param>
        public ReduceScatterOperation(IProcessGroup processGroup, long[] fullShape, DataType dataType, int shardIndex, ReduceOp reduceOp = ReduceOp.Sum)
        {
            _processGroup = processGroup ?? throw new ArgumentNullException(nameof(processGroup));
            _shardIndex = shardIndex;
            _reduceOp = reduceOp;

            // Calculate total size
            long totalSize = 1;
            foreach (var dim in fullShape)
                totalSize *= dim;

            // Calculate shard size
            var worldSize = _processGroup.WorldSize;
            var shardSize = (totalSize + worldSize - 1) / worldSize;

            // Allocate buffer for scattered result (only our shard)
            var intShape = new int[] { (int)shardSize };
            _shardedBuffer = Tensor.Zeros(intShape, dataType);
        }

        /// <summary>
        /// Perform Reduce-Scatter: Reduce all tensors and scatter the result.
        /// Each device gets a reduced portion of the result.
        /// </summary>
        /// <param name="fullTensor">Full tensor to reduce and scatter</param>
        /// <returns>Reduced shard owned by this device</returns>
        public Tensor ReduceScatter(Tensor fullTensor)
        {
            if (fullTensor == null)
                throw new ArgumentNullException(nameof(fullTensor));

            var worldSize = _processGroup.WorldSize;
            var rank = _processGroup.Rank;

            // Edge case: single device
            if (worldSize == 1)
            {
                // No need to scatter, just copy
                Array.Copy(fullTensor.Data, _shardedBuffer.Data, Math.Min(fullTensor.Size, _shardedBuffer.Size));
                return _shardedBuffer;
            }

            // Calculate chunk size for each rank
            var totalSize = fullTensor.Size;
            var chunkSize = (totalSize + worldSize - 1) / worldSize;

            // Phase 1: Reduce-Scatter using ring algorithm
            // Each device reduces its portion and sends to the next device
            for (int step = 0; step < worldSize - 1; step++)
            {
                var sendTo = (rank + 1) % worldSize;
                var recvFrom = (rank - 1 + worldSize) % worldSize;

                // Calculate which chunk to send and receive
                var sendChunkIndex = (rank - step + worldSize) % worldSize;
                var recvChunkIndex = (rank - step - 1 + worldSize) % worldSize;

                // Get chunk to send
                var sendOffset = sendChunkIndex * chunkSize;
                var sendSize = Math.Min(chunkSize, totalSize - sendOffset);
                var sendData = new float[sendSize];
                Array.Copy(fullTensor.Data, sendOffset, sendData, 0, sendSize);
                var sendTensor = Tensor.FromArray(sendData);

                // Get buffer for receive
                var recvOffset = recvChunkIndex * chunkSize;
                var recvSize = Math.Min(chunkSize, totalSize - recvOffset);
                var recvData = new float[recvSize];
                var recvTensor = Tensor.FromArray(recvData);

                // Concurrent send and receive
                var sendTask = _processGroup.SendAsync(sendTensor, sendTo);
                var recvTask = _processGroup.RecvAsync(recvTensor, recvFrom);
                Task.WaitAll(sendTask, recvTask);

                // Reduce received data with local data
                ReduceData(fullTensor.Data, recvData, recvOffset, recvSize);

                // Clean up
                // Note: Tensor doesn't implement IDisposable, so we let GC handle these
            }

            // Phase 2: Extract our shard from the reduced result
            var myShardOffset = _shardIndex * chunkSize;
            var myShardSize = Math.Min(chunkSize, totalSize - myShardOffset);
            Array.Copy(fullTensor.Data, myShardOffset, _shardedBuffer.Data, 0, Math.Min(myShardSize, _shardedBuffer.Size));

            // Handle Avg operation
            if (_reduceOp == ReduceOp.Avg)
            {
                for (int i = 0; i < _shardedBuffer.Size; i++)
                {
                    _shardedBuffer.Data[i] /= worldSize;
                }
            }

            return _shardedBuffer;
        }

        /// <summary>
        /// Reduce received data with local data in-place.
        /// </summary>
        private void ReduceData(float[] localData, float[] receivedData, int offset, int size)
        {
            switch (_reduceOp)
            {
                case ReduceOp.Sum:
                case ReduceOp.Avg:
                    for (int i = 0; i < size; i++)
                    {
                        localData[offset + i] += receivedData[i];
                    }
                    break;

                case ReduceOp.Product:
                    for (int i = 0; i < size; i++)
                    {
                        localData[offset + i] *= receivedData[i];
                    }
                    break;

                case ReduceOp.Max:
                    for (int i = 0; i < size; i++)
                    {
                        localData[offset + i] = Math.Max(localData[offset + i], receivedData[i]);
                    }
                    break;

                case ReduceOp.Min:
                    for (int i = 0; i < size; i++)
                    {
                        localData[offset + i] = Math.Min(localData[offset + i], receivedData[i]);
                    }
                    break;

                default:
                    throw new ArgumentException($"Unsupported reduction operation: {_reduceOp}", nameof(_reduceOp));
            }
        }

        /// <summary>
        /// Get the reduction operation.
        /// </summary>
        /// <returns>The reduce operation type</returns>
        public ReduceOp GetReduceOp()
        {
            return _reduceOp;
        }

        /// <summary>
        /// Perform asynchronous Reduce-Scatter.
        /// </summary>
        /// <param name="fullTensor">Full tensor to reduce and scatter</param>
        /// <returns>Task that completes with the reduced shard</returns>
        public Task<Tensor> ReduceScatterAsync(Tensor fullTensor)
        {
            return Task.Run(() => ReduceScatter(fullTensor));
        }

        /// <summary>
        /// Dispose of resources.
        /// </summary>
        public void Dispose()
        {
            // _shardedBuffer doesn't implement IDisposable, so no explicit cleanup needed
            // The buffer will be garbage collected automatically
        }
    }
}
