using System;
using System.Collections.Concurrent;
using System.Threading;

namespace MLFramework.Data.Worker
{
    /// <summary>
    /// High-performance shared memory queue for efficient communication between workers and main process.
    /// </summary>
    /// <typeparam name="T">The type of items in the queue.</typeparam>
    public class SharedMemoryQueue<T> : ISharedMemoryQueue<T>
    {
        private readonly ConcurrentQueue<T> _queue;
        private readonly SemaphoreSlim _enqueueSemaphore;
        private readonly SemaphoreSlim _dequeueSemaphore;
        private volatile bool _isDisposed;
        private readonly int _maxSize;

        /// <summary>
        /// Initializes a new instance of the SharedMemoryQueue class.
        /// </summary>
        /// <param name="maxSize">Maximum number of items in the queue. 0 means unbounded.</param>
        public SharedMemoryQueue(int maxSize = 0)
        {
            if (maxSize < 0)
                throw new ArgumentOutOfRangeException(nameof(maxSize), "maxSize must be non-negative");

            _queue = new ConcurrentQueue<T>();
            _maxSize = maxSize;

            if (maxSize > 0)
            {
                _enqueueSemaphore = new SemaphoreSlim(maxSize, maxSize);
                _dequeueSemaphore = new SemaphoreSlim(0, maxSize);
            }
            else
            {
                // Unbounded queue
                _enqueueSemaphore = null;
                _dequeueSemaphore = null;
            }

            _isDisposed = false;
        }

        /// <inheritdoc/>
        public void Enqueue(T item)
        {
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(SharedMemoryQueue<T>));

            if (_maxSize > 0)
            {
                // Wait if queue is full
                _enqueueSemaphore.Wait();
            }

            _queue.Enqueue(item);

            if (_maxSize > 0)
            {
                _dequeueSemaphore.Release();
            }
        }

        /// <inheritdoc/>
        public T Dequeue()
        {
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(SharedMemoryQueue<T>));

            if (_maxSize > 0)
            {
                // Wait if queue is empty
                _dequeueSemaphore.Wait();
            }

            if (_queue.TryDequeue(out var item))
            {
                if (_maxSize > 0)
                {
                    _enqueueSemaphore.Release();
                }

                return item;
            }

            throw new InvalidOperationException("Queue is empty");
        }

        /// <inheritdoc/>
        public bool TryDequeue(out T item)
        {
            if (_isDisposed)
            {
                item = default;
                return false;
            }

            if (_maxSize > 0)
            {
                if (!_dequeueSemaphore.Wait(0))
                {
                    item = default;
                    return false;
                }
            }

            if (_queue.TryDequeue(out item))
            {
                if (_maxSize > 0)
                {
                    _enqueueSemaphore.Release();
                }

                return true;
            }

            if (_maxSize > 0)
            {
                _dequeueSemaphore.Release();
            }

            return false;
        }

        /// <inheritdoc/>
        public int Count => _queue.Count;

        /// <inheritdoc/>
        public bool IsEmpty => _queue.IsEmpty;

        /// <inheritdoc/>
        public void Clear()
        {
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(SharedMemoryQueue<T>));

            while (_queue.TryDequeue(out _))
            {
                if (_maxSize > 0)
                {
                    _enqueueSemaphore.Release();
                }
            }
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_isDisposed)
                return;

            _isDisposed = true;

            // Unblock any waiting operations
            if (_maxSize > 0)
            {
                _enqueueSemaphore?.Dispose();
                _dequeueSemaphore?.Dispose();
            }

            // Clear queue
            Clear();
        }
    }
}
