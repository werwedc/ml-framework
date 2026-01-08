using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;

namespace MLFramework.Data.Memory
{
    /// <summary>
    /// Specialized pool for pinned memory buffers.
    /// Efficiently reuses pinned buffers to reduce allocation overhead and garbage collection.
    /// </summary>
    /// <typeparam name="T">The type of elements in the buffers. Must be unmanaged.</typeparam>
    public sealed class PinnedMemoryPool<T> : IDisposable
        where T : unmanaged
    {
        private int _bufferSize;
        private readonly int _maxSize;
        private readonly ConcurrentQueue<PinnedBuffer<T>> _pool;
        private bool _isDisposed;

        /// <summary>
        /// Initializes a new instance of the PinnedMemoryPool class.
        /// </summary>
        /// <param name="bufferSize">Size of each buffer in the pool (number of elements).</param>
        /// <param name="initialSize">Number of buffers to pre-allocate. Default is 0.</param>
        /// <param name="maxSize">Maximum number of buffers to keep in the pool. Default is 20.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when bufferSize or maxSize is less than or equal to zero.</exception>
        public PinnedMemoryPool(int bufferSize, int initialSize = 0, int maxSize = 20)
        {
            if (bufferSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(bufferSize), "Buffer size must be greater than zero.");

            if (maxSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxSize), "Max size must be greater than zero.");

            _bufferSize = bufferSize;
            _maxSize = maxSize;
            _pool = new ConcurrentQueue<PinnedBuffer<T>>();

            // Pre-allocate initial buffers
            for (int i = 0; i < initialSize; i++)
            {
                _pool.Enqueue(PinnedBuffer<T>.Allocate(_bufferSize));
            }
        }

        /// <summary>
        /// Gets the size of each buffer in the pool (number of elements).
        /// </summary>
        public int BufferSize => _bufferSize;

        /// <summary>
        /// Gets the maximum number of buffers the pool can hold.
        /// </summary>
        public int MaxSize => _maxSize;

        /// <summary>
        /// Gets the current number of buffers in the pool.
        /// </summary>
        public int Count => _pool.Count;

        /// <summary>
        /// Rents a pinned buffer from the pool.
        /// Returns an available pinned buffer from the pool or creates a new one if the pool is empty.
        /// All buffers are already pinned and ready for GPU transfer.
        /// </summary>
        /// <returns>A pinned buffer ready for use.</returns>
        public PinnedBuffer<T> Rent()
        {
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(PinnedMemoryPool<T>));

            // Try to get a buffer from the pool
            if (_pool.TryDequeue(out var buffer))
            {
                return buffer;
            }

            // Pool is empty, create a new buffer
            return PinnedBuffer<T>.Allocate(_bufferSize);
        }

        /// <summary>
        /// Returns a pinned buffer to the pool for reuse.
        /// </summary>
        /// <param name="buffer">The buffer to return to the pool.</param>
        /// <exception cref="ObjectDisposedException">Thrown when the pool is disposed.</exception>
        /// <exception cref="ArgumentNullException">Thrown when buffer is null.</exception>
        /// <exception cref="ArgumentException">Thrown when buffer size doesn't match pool configuration.</exception>
        public void Return(PinnedBuffer<T> buffer)
        {
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(PinnedMemoryPool<T>));

            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));

            // Validate buffer size matches pool configuration
            if (buffer.Length != _bufferSize)
                throw new ArgumentException("Buffer size doesn't match pool configuration.", nameof(buffer));

            // Return to pool if under max size, otherwise let it be garbage collected
            if (_pool.Count < _maxSize)
            {
                // Optional: Clear buffer contents before returning to pool
                // buffer.Fill(default);

                _pool.Enqueue(buffer);
            }
            else
            {
                // Pool is at max capacity, dispose the buffer
                buffer.Dispose();
            }
        }

        /// <summary>
        /// Resizes the pool to use buffers of a new size.
        /// Clears existing pool and allocates buffers of the new size.
        /// Useful for changing batch sizes dynamically.
        /// </summary>
        /// <param name="newBufferSize">The new buffer size (number of elements).</param>
        /// <exception cref="ObjectDisposedException">Thrown when the pool is disposed.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when newBufferSize is less than or equal to zero.</exception>
        public void Resize(int newBufferSize)
        {
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(PinnedMemoryPool<T>));

            if (newBufferSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(newBufferSize), "Buffer size must be greater than zero.");

            // Clear existing pool
            Clear();

            // Update buffer size
            _bufferSize = newBufferSize;
        }

        /// <summary>
        /// Clears all buffers from the pool and disposes them.
        /// </summary>
        public void Clear()
        {
            while (_pool.TryDequeue(out var buffer))
            {
                buffer.Dispose();
            }
        }

        /// <summary>
        /// Disposes the pool and all contained buffers.
        /// </summary>
        public void Dispose()
        {
            if (!_isDisposed)
            {
                Clear();
                _isDisposed = true;
            }
        }
    }
}
