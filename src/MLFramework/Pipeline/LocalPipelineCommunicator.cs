using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using RitterFramework.Core.Tensor;

namespace MLFramework.Pipeline
{
    /// <summary>
    /// Direction of communication in pipeline
    /// </summary>
    public enum CommunicationDirection
    {
        /// <summary>
        /// Forward pass: sending activations to next stage
        /// </summary>
        Forward,

        /// <summary>
        /// Backward pass: sending gradients to previous stage
        /// </summary>
        Backward
    }

    /// <summary>
    /// In-memory communicator for single-device testing (no actual network)
    /// Uses shared memory buffers for communication
    /// </summary>
    public class LocalPipelineCommunicator : IPipelineCommunicator
    {
        private readonly int _rank;
        private readonly int _worldSize;
        private readonly ConcurrentDictionary<(int from, int to), BlockingCollection<Tensor?>> _buffers;
        private readonly SemaphoreSlim _barrier;
        private int _disposed;
        private static readonly int _timeoutMs = 30000; // 30 seconds timeout

        /// <summary>
        /// Rank of the current process/device
        /// </summary>
        public int Rank => _rank;

        /// <summary>
        /// Total number of processes/devices
        /// </summary>
        public int WorldSize => _worldSize;

        public LocalPipelineCommunicator(int rank, int worldSize)
        {
            if (rank < 0 || rank >= worldSize)
                throw new ArgumentOutOfRangeException(nameof(rank), $"Rank must be in [0, {worldSize - 1}]");
            if (worldSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(worldSize), "World size must be positive");

            _rank = rank;
            _worldSize = worldSize;
            _buffers = new ConcurrentDictionary<(int, int), BlockingCollection<Tensor?>>();

            // Create buffers for all possible communication pairs
            for (int from = 0; from < worldSize; from++)
            {
                for (int to = 0; to < worldSize; to++)
                {
                    if (from != to)
                    {
                        _buffers[(from, to)] = new BlockingCollection<Tensor?>(boundedCapacity: 10);
                    }
                }
            }

            // Initialize barrier with count = worldSize - 1
            _barrier = new SemaphoreSlim(worldSize - 1, worldSize - 1);
        }

        /// <summary>
        /// Send a tensor to a specific stage
        /// </summary>
        public async Task SendAsync(Tensor tensor, int destRank)
        {
            ThrowIfDisposed();

            if (destRank < 0 || destRank >= _worldSize)
                throw new ArgumentOutOfRangeException(nameof(destRank), $"Destination rank must be in [0, {_worldSize - 1}]");
            if (destRank == _rank)
                throw new ArgumentException("Cannot send to self", nameof(destRank));

            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            var buffer = _buffers[(_rank, destRank)];
            await Task.Run(() => buffer.Add(tensor));
        }

        /// <summary>
        /// Receive a tensor from a specific stage
        /// </summary>
        public async Task<Tensor> ReceiveAsync(int srcRank)
        {
            ThrowIfDisposed();

            if (srcRank < 0 || srcRank >= _worldSize)
                throw new ArgumentOutOfRangeException(nameof(srcRank), $"Source rank must be in [0, {_worldSize - 1}]");
            if (srcRank == _rank)
                throw new ArgumentException("Cannot receive from self", nameof(srcRank));

            var buffer = _buffers[(srcRank, _rank)];
            var tensor = await Task.Run(() =>
            {
                if (buffer.TryTake(out var result, _timeoutMs))
                {
                    return result;
                }
                throw new TimeoutException($"Receive timeout waiting for rank {srcRank}");
            });

            if (tensor == null)
                throw new InvalidOperationException("Received null tensor (sentinel)");

            return tensor;
        }

        /// <summary>
        /// Broadcast a tensor from root to all stages
        /// </summary>
        public async Task<Tensor> BroadcastAsync(Tensor tensor, int root)
        {
            ThrowIfDisposed();

            if (root < 0 || root >= _worldSize)
                throw new ArgumentOutOfRangeException(nameof(root), $"Root rank must be in [0, {_worldSize - 1}]");

            if (_rank == root)
            {
                // Root sends to all other ranks
                var tasks = new List<Task>();
                for (int dest = 0; dest < _worldSize; dest++)
                {
                    if (dest != root)
                    {
                        tasks.Add(SendAsync(tensor, dest));
                    }
                }
                await Task.WhenAll(tasks);
                return tensor;
            }
            else
            {
                // Non-root ranks receive from root
                return await ReceiveAsync(root);
            }
        }

        /// <summary>
        /// Synchronize all stages (barrier)
        /// </summary>
        public async Task BarrierAsync()
        {
            ThrowIfDisposed();

            // Use a simple barrier: each rank waits until all have arrived
            await Task.Run(() =>
            {
                // Release all semaphore slots
                for (int i = 0; i < _worldSize - 1; i++)
                {
                    _barrier.Release();
                }

                // Wait for all other ranks to do the same
                for (int i = 0; i < _worldSize - 1; i++)
                {
                    if (!_barrier.Wait(_timeoutMs))
                    {
                        throw new TimeoutException("Barrier timeout");
                    }
                }
            });
        }

        private void ThrowIfDisposed()
        {
            if (Interlocked.CompareExchange(ref _disposed, 0, 0) == 1)
                throw new ObjectDisposedException(nameof(LocalPipelineCommunicator));
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 1)
                return;

            _barrier?.Dispose();

            foreach (var buffer in _buffers.Values)
            {
                buffer.CompleteAdding();
                buffer.Dispose();
            }

            _buffers.Clear();
        }
    }
}
