using RitterFramework.Core.Tensor;
using System;
using System.Threading.Tasks;

namespace MLFramework.Distributed
{
    /// <summary>
    /// Implements the Ring-AllReduce algorithm for bandwidth-efficient gradient aggregation.
    /// This algorithm uses a ring topology to minimize communication overhead.
    /// </summary>
    public class RingAllReduce
    {
        private readonly IProcessGroup _processGroup;

        /// <summary>
        /// Initializes a new instance of the RingAllReduce class.
        /// </summary>
        /// <param name="processGroup">The process group to use for communication.</param>
        public RingAllReduce(IProcessGroup processGroup)
        {
            _processGroup = processGroup ?? throw new ArgumentNullException(nameof(processGroup));
        }

        /// <summary>
        /// Performs Ring-AllReduce on the given tensor.
        /// Modifies the tensor in-place with the reduced result.
        /// </summary>
        /// <param name="tensor">The tensor to reduce.</param>
        /// <param name="op">The reduction operation to perform.</param>
        public void AllReduce(Tensor tensor, ReduceOp op = ReduceOp.Sum)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            // Handle edge cases
            if (_processGroup.WorldSize == 1)
            {
                // Single device - no communication needed
                return;
            }

            if (tensor.Size == 0)
            {
                // Empty tensor - nothing to do
                return;
            }

            // Ensure tensor is contiguous
            var contiguousTensor = tensor;

            // Phase 1: Reduce-Scatter
            ReduceScatter(contiguousTensor, op);

            // Phase 2: AllGather
            AllGather(contiguousTensor);

            // Handle Avg operation
            if (op == ReduceOp.Avg)
            {
                var worldSize = _processGroup.WorldSize;
                for (int i = 0; i < contiguousTensor.Size; i++)
                {
                    contiguousTensor.Data[i] /= worldSize;
                }
            }
        }

        /// <summary>
        /// Performs asynchronous Ring-AllReduce on the given tensor.
        /// </summary>
        /// <param name="tensor">The tensor to reduce.</param>
        /// <param name="op">The reduction operation to perform.</param>
        /// <returns>A task that completes when the operation is done.</returns>
        public Task AllReduceAsync(Tensor tensor, ReduceOp op = ReduceOp.Sum)
        {
            return Task.Run(() => AllReduce(tensor, op));
        }

        /// <summary>
        /// Reduce-Scatter phase: Each device ends up with a portion of the reduced result.
        /// </summary>
        /// <param name="tensor">The tensor to reduce.</param>
        /// <param name="op">The reduction operation.</param>
        private void ReduceScatter(Tensor tensor, ReduceOp op)
        {
            var rank = _processGroup.Rank;
            var worldSize = _processGroup.WorldSize;
            var chunkManager = new ChunkManager(tensor, worldSize);
            var numChunks = chunkManager.NumChunks;

            // Get the chunk this rank is responsible for reducing
            var myChunkIndex = chunkManager.GetChunkForRank(rank);
            var myChunk = chunkManager.GetChunk(myChunkIndex);

            // In each step, send our current chunk and receive a new one
            for (int step = 0; step < worldSize - 1; step++)
            {
                // Calculate send and receive ranks
                var sendTo = (rank + 1) % worldSize;
                var recvFrom = (rank - 1 + worldSize) % worldSize;

                // Calculate which chunk to send and receive
                var sendChunkIndex = (myChunkIndex - step + worldSize) % worldSize;
                var recvChunkIndex = (myChunkIndex - step - 1 + worldSize) % worldSize;

                // Get chunks to send and receive
                var sendChunk = chunkManager.GetChunk(sendChunkIndex);
                var recvChunk = chunkManager.GetChunk(recvChunkIndex);

                // Send and receive concurrently
                var sendTask = _processGroup.SendAsync(sendChunk, sendTo);
                var recvTask = _processGroup.RecvAsync(recvChunk, recvFrom);

                // Wait for both operations to complete
                Task.WaitAll(sendTask, recvTask);

                // Reduce the received chunk with our local chunk
                ReduceChunks(myChunk, recvChunk, op);
            }
        }

        /// <summary>
        /// AllGather phase: Each device gathers all portions of the reduced result.
        /// </summary>
        /// <param name="tensor">The tensor to gather.</param>
        private void AllGather(Tensor tensor)
        {
            var rank = _processGroup.Rank;
            var worldSize = _processGroup.WorldSize;
            var chunkManager = new ChunkManager(tensor, worldSize);
            var numChunks = chunkManager.NumChunks;

            // Get the chunk this rank has reduced
            var myChunkIndex = chunkManager.GetChunkForRank(rank);
            var myChunk = chunkManager.GetChunk(myChunkIndex);

            // In each step, send our reduced chunk and receive a new chunk
            for (int step = 0; step < worldSize - 1; step++)
            {
                // Calculate send and receive ranks
                var sendTo = (rank + 1) % worldSize;
                var recvFrom = (rank - 1 + worldSize) % worldSize;

                // Calculate which chunk to send and receive
                var sendChunkIndex = (myChunkIndex + step) % worldSize;
                var recvChunkIndex = (myChunkIndex + step + 1) % worldSize;

                // Get chunks to send and receive
                var sendChunk = chunkManager.GetChunk(sendChunkIndex);
                var recvChunk = chunkManager.GetChunk(recvChunkIndex);

                // Send and receive concurrently
                var sendTask = _processGroup.SendAsync(sendChunk, sendTo);
                var recvTask = _processGroup.RecvAsync(recvChunk, recvFrom);

                // Wait for both operations to complete
                Task.WaitAll(sendTask, recvTask);
            }
        }

        /// <summary>
        /// Reduces two chunks element-wise using the specified operation.
        /// </summary>
        /// <param name="dest">The destination chunk (modified in-place).</param>
        /// <param name="src">The source chunk.</param>
        /// <param name="op">The reduction operation.</param>
        private void ReduceChunks(Tensor dest, Tensor src, ReduceOp op)
        {
            var destData = dest.Data;
            var srcData = src.Data;
            var length = Math.Min(dest.Size, src.Size);

            switch (op)
            {
                case ReduceOp.Sum:
                    for (int i = 0; i < length; i++)
                    {
                        destData[i] += srcData[i];
                    }
                    break;

                case ReduceOp.Product:
                    for (int i = 0; i < length; i++)
                    {
                        destData[i] *= srcData[i];
                    }
                    break;

                case ReduceOp.Max:
                    for (int i = 0; i < length; i++)
                    {
                        destData[i] = Math.Max(destData[i], srcData[i]);
                    }
                    break;

                case ReduceOp.Min:
                    for (int i = 0; i < length; i++)
                    {
                        destData[i] = Math.Min(destData[i], srcData[i]);
                    }
                    break;

                case ReduceOp.Avg:
                    // Avg is handled at the end of AllReduce
                    // During ReduceScatter, we just sum
                    for (int i = 0; i < length; i++)
                    {
                        destData[i] += srcData[i];
                    }
                    break;

                default:
                    throw new ArgumentException($"Unsupported reduction operation: {op}", nameof(op));
            }
        }
    }
}
