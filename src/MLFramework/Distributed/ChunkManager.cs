using RitterFramework.Core.Tensor;
using System;
using System.Linq;

namespace MLFramework.Distributed
{
    /// <summary>
    /// Helper class to divide tensors into chunks for ring communication.
    /// </summary>
    internal class ChunkManager
    {
        private readonly Tensor _tensor;
        private readonly int _numChunks;
        private readonly long[] _chunkSizes;
        private readonly long[] _chunkOffsets;

        /// <summary>
        /// Initializes a new instance of the ChunkManager class.
        /// </summary>
        /// <param name="tensor">The tensor to chunk.</param>
        /// <param name="numChunks">The number of chunks to divide the tensor into.</param>
        public ChunkManager(Tensor tensor, int numChunks)
        {
            if (tensor == null)
            {
                throw new ArgumentNullException(nameof(tensor));
            }

            if (numChunks <= 0)
            {
                throw new ArgumentException("Number of chunks must be positive", nameof(numChunks));
            }

            _tensor = tensor;
            _numChunks = numChunks;
            _chunkSizes = new long[numChunks];
            _chunkOffsets = new long[numChunks];

            CalculateChunkSizes();
        }

        /// <summary>
        /// Gets the number of chunks.
        /// </summary>
        public int NumChunks => _numChunks;

        /// <summary>
        /// Gets a specific chunk from the tensor.
        /// </summary>
        /// <param name="chunkIndex">The index of the chunk to get.</param>
        /// <returns>A tensor containing the chunk data.</returns>
        public Tensor GetChunk(int chunkIndex)
        {
            if (chunkIndex < 0 || chunkIndex >= _numChunks)
            {
                throw new ArgumentOutOfRangeException(nameof(chunkIndex),
                    $"Chunk index must be between 0 and {_numChunks - 1}");
            }

            var chunkSize = _chunkSizes[chunkIndex];
            var chunkOffset = _chunkOffsets[chunkIndex];

            // Create a new tensor for this chunk
            var chunkData = new float[chunkSize];
            Array.Copy(_tensor.Data, chunkOffset, chunkData, 0, chunkSize);

            // Create a 1D tensor for the chunk
            var chunk = new Tensor(chunkData, new int[] { (int)chunkSize }, false, _tensor.Dtype);
            return chunk;
        }

        /// <summary>
        /// Sets a specific chunk in the tensor.
        /// </summary>
        /// <param name="chunkIndex">The index of the chunk to set.</param>
        /// <param name="chunk">The chunk tensor to copy data from.</param>
        public void SetChunk(int chunkIndex, Tensor chunk)
        {
            if (chunkIndex < 0 || chunkIndex >= _numChunks)
            {
                throw new ArgumentOutOfRangeException(nameof(chunkIndex),
                    $"Chunk index must be between 0 and {_numChunks - 1}");
            }

            if (chunk == null)
            {
                throw new ArgumentNullException(nameof(chunk));
            }

            var chunkSize = _chunkSizes[chunkIndex];
            var chunkOffset = _chunkOffsets[chunkIndex];

            if (chunk.Size != chunkSize)
            {
                throw new ArgumentException(
                    $"Chunk size mismatch. Expected {chunkSize}, got {chunk.Size}",
                    nameof(chunk));
            }

            // Copy chunk data back to the original tensor
            Array.Copy(chunk.Data, 0, _tensor.Data, chunkOffset, chunkSize);
        }

        /// <summary>
        /// Gets the chunk index that belongs to a given rank.
        /// </summary>
        /// <param name="rank">The rank to get the chunk for.</param>
        /// <returns>The chunk index for the given rank.</returns>
        public int GetChunkForRank(int rank)
        {
            if (rank < 0 || rank >= _numChunks)
            {
                throw new ArgumentOutOfRangeException(nameof(rank),
                    $"Rank must be between 0 and {_numChunks - 1}");
            }

            return rank;
        }

        /// <summary>
        /// Calculates the size and offset for each chunk.
        /// Handles uneven division by giving the last chunk the remaining elements.
        /// </summary>
        private void CalculateChunkSizes()
        {
            var totalElements = (long)_tensor.Size;
            var baseChunkSize = totalElements / _numChunks;
            var remainder = totalElements % _numChunks;

            for (int i = 0; i < _numChunks; i++)
            {
                _chunkSizes[i] = baseChunkSize;
                if (i < remainder)
                {
                    _chunkSizes[i]++;
                }
            }

            // Calculate offsets
            _chunkOffsets[0] = 0;
            for (int i = 1; i < _numChunks; i++)
            {
                _chunkOffsets[i] = _chunkOffsets[i - 1] + _chunkSizes[i - 1];
            }
        }
    }
}
