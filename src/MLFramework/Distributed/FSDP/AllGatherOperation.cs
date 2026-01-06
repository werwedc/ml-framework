using RitterFramework.Core;
using RitterFramework.Core.Tensor;
using MLFramework.Distributed;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Implements All-Gather communication operation for FSDP.
    /// Gathers parameter shards from all devices onto every device.
    /// </summary>
    public class AllGatherOperation : IDisposable
    {
        private readonly IProcessGroup _processGroup;
        private readonly Tensor _gatheredBuffer;
        private readonly bool _isOwner;

        /// <summary>
        /// Initialize a new All-Gather operation.
        /// </summary>
        /// <param name="processGroup">Process group for communication</param>
        /// <param name="fullShape">Shape of the fully gathered tensor</param>
        /// <param name="dataType">Data type of the tensor</param>
        /// <param name="shardIndex">Index of the shard on this device</param>
        public AllGatherOperation(IProcessGroup processGroup, long[] fullShape, DataType dataType, int shardIndex)
        {
            _processGroup = processGroup ?? throw new ArgumentNullException(nameof(processGroup));

            // Calculate total size
            long totalSize = 1;
            foreach (var dim in fullShape)
                totalSize *= dim;

            // Calculate shard size
            var worldSize = _processGroup.WorldSize;
            var shardSize = (totalSize + worldSize - 1) / worldSize;

            // Convert long[] shape to int[] for Tensor creation
            var intShape = new int[fullShape.Length];
            for (int i = 0; i < fullShape.Length; i++)
            {
                intShape[i] = (int)fullShape[i];
            }

            // Allocate buffer for gathered result
            _gatheredBuffer = Tensor.Zeros(intShape, dataType);

            // Determine if this device is the owner (first device owns first shard, etc.)
            _isOwner = (shardIndex == processGroup.Rank);
        }

        /// <summary>
        /// Perform All-Gather: Collect all shards from all devices.
        /// Each device gets the full result.
        /// </summary>
        /// <param name="shardedTensor">Local shard to contribute</param>
        /// <returns>Full tensor containing all shards</returns>
        public Tensor AllGather(Tensor shardedTensor)
        {
            if (shardedTensor == null)
                throw new ArgumentNullException(nameof(shardedTensor));

            var worldSize = _processGroup.WorldSize;
            var rank = _processGroup.Rank;

            // Edge case: single device
            if (worldSize == 1)
            {
                // No need to gather, just copy
                Array.Copy(shardedTensor.Data, _gatheredBuffer.Data, shardedTensor.Size);
                return _gatheredBuffer;
            }

            // Calculate chunk size for each rank
            var totalSize = _gatheredBuffer.Size;
            var chunkSize = (totalSize + worldSize - 1) / worldSize;

            // Phase 1: Each device sends its shard to all others
            // We'll use point-to-point communication for this
            for (int step = 0; step < worldSize; step++)
            {
                var srcRank = step;
                var dstRank = (rank + 1) % worldSize;

                if (step == 0)
                {
                    // First step: copy our own shard
                    var startOffset = rank * chunkSize;
                    var sizeToCopy = Math.Min(chunkSize, totalSize - startOffset);
                    Array.Copy(shardedTensor.Data, 0, _gatheredBuffer.Data, startOffset, sizeToCopy);
                }
                else
                {
                    // Receive shard from another device
                    var recvOffset = srcRank * chunkSize;
                    var recvSize = Math.Min(chunkSize, totalSize - recvOffset);
                    var recvBuffer = new float[recvSize];

                    // Create a tensor for the received data
                    var recvShape = new int[] { (int)recvSize };
                    var recvTensor = new Tensor(recvBuffer, recvShape);

                    // Receive from source rank
                    _processGroup.Recv(recvTensor, srcRank);

                    // Copy received shard to buffer
                    Array.Copy(recvBuffer, 0, _gatheredBuffer.Data, recvOffset, recvSize);

                    // recvTensor will be garbage collected automatically
                }
            }

            return _gatheredBuffer;
        }

        /// <summary>
        /// Perform asynchronous All-Gather.
        /// </summary>
        /// <param name="shardedTensor">Local shard to contribute</param>
        /// <returns>Task that completes with the full gathered tensor</returns>
        public Task<Tensor> AllGatherAsync(Tensor shardedTensor)
        {
            return Task.Run(() => AllGather(shardedTensor));
        }

        /// <summary>
        /// Dispose of resources.
        /// </summary>
        public void Dispose()
        {
            // _gatheredBuffer will be garbage collected automatically
            // (Tensor class doesn't implement IDisposable)
        }
    }
}
