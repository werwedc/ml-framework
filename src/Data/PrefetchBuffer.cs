using System.Collections.Concurrent;

namespace MLFramework.Data
{
    /// <summary>
    /// Internal buffer that holds prefetched items in FIFO order.
    /// Thread-safe implementation using ConcurrentQueue.
    /// </summary>
    /// <typeparam name="T">The type of items in the buffer.</typeparam>
    public class PrefetchBuffer<T>
    {
        private readonly ConcurrentQueue<T> _buffer;
        private readonly int _capacity;
        private readonly object _lock = new object();

        /// <summary>
        /// Initializes a new instance of the PrefetchBuffer class.
        /// </summary>
        /// <param name="capacity">Maximum number of items the buffer can hold.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when capacity is less than or equal to zero.</exception>
        public PrefetchBuffer(int capacity)
        {
            if (capacity <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity must be greater than zero.");
            }

            _buffer = new ConcurrentQueue<T>();
            _capacity = capacity;
        }

        /// <summary>
        /// Gets the current number of items in the buffer.
        /// </summary>
        public int Count
        {
            get
            {
                lock (_lock)
                {
                    return _buffer.Count;
                }
            }
        }

        /// <summary>
        /// Gets the maximum capacity of the buffer.
        /// </summary>
        public int Capacity => _capacity;

        /// <summary>
        /// Gets whether the buffer is empty.
        /// </summary>
        public bool IsEmpty => Count == 0;

        /// <summary>
        /// Gets whether the buffer is full.
        /// </summary>
        public bool IsFull => Count >= _capacity;

        /// <summary>
        /// Adds an item to the buffer.
        /// </summary>
        /// <param name="item">The item to add.</param>
        /// <exception cref="InvalidOperationException">Thrown when buffer is full.</exception>
        public void Add(T item)
        {
            if (IsFull)
            {
                throw new InvalidOperationException("Cannot add item to full prefetch buffer.");
            }

            _buffer.Enqueue(item);
        }

        /// <summary>
        /// Removes and returns the next item from the buffer in FIFO order.
        /// </summary>
        /// <returns>The next item in the buffer.</returns>
        /// <exception cref="InvalidOperationException">Thrown when buffer is empty.</exception>
        public T GetNext()
        {
            if (_buffer.TryDequeue(out T item))
            {
                return item;
            }

            throw new InvalidOperationException("Cannot get item from empty prefetch buffer.");
        }

        /// <summary>
        /// Returns the next item without removing it from the buffer.
        /// </summary>
        /// <returns>The next item in the buffer.</returns>
        /// <exception cref="InvalidOperationException">Thrown when buffer is empty.</exception>
        public T Peek()
        {
            if (_buffer.TryPeek(out T item))
            {
                return item;
            }

            throw new InvalidOperationException("Cannot peek at empty prefetch buffer.");
        }

        /// <summary>
        /// Tries to get the next item from the buffer.
        /// </summary>
        /// <param name="item">When this method returns, contains the item if found, otherwise default value.</param>
        /// <returns>True if an item was found, false if buffer is empty.</returns>
        public bool TryGet(out T item)
        {
            return _buffer.TryDequeue(out item);
        }

        /// <summary>
        /// Clears all items from the buffer.
        /// </summary>
        public void Clear()
        {
            while (_buffer.TryDequeue(out _)) { }
        }
    }
}
