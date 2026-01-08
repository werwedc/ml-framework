namespace MLFramework.Data
{
    /// <summary>
    /// Thread-safe blocking queue for communication between producers and consumers.
    /// Built on top of BlockingCollection for efficient producer-consumer pattern.
    /// </summary>
    /// <typeparam name="T">The type of items in the queue.</typeparam>
    public class SharedQueue<T> : IDisposable
    {
        private readonly BlockingCollection<T> _collection;
        private readonly QueueStatistics _statistics;
        private readonly bool _enableStatistics;
        private readonly object _statisticsLock = new object();

        /// <summary>
        /// Initializes a new instance of the SharedQueue class with the specified capacity.
        /// </summary>
        /// <param name="capacity">Maximum number of items in the queue (must be > 0).</param>
        /// <param name="cancellationToken">Optional cancellation token for graceful shutdown.</param>
        /// <param name="enableStatistics">Whether to enable statistics tracking (default: true).</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when capacity is less than or equal to zero.</exception>
        public SharedQueue(int capacity, CancellationToken? cancellationToken = null, bool enableStatistics = true)
        {
            if (capacity <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(capacity), "Capacity must be greater than zero.");
            }

            _collection = new BlockingCollection<T>(capacity);
            _enableStatistics = enableStatistics;
            _statistics = new QueueStatistics();

            if (cancellationToken.HasValue)
            {
                cancellationToken.Value.Register(() =>
                {
                    try
                    {
                        _collection.CancelPendingConsumers();
                        _collection.CancelPendingProducers();
                    }
                    catch (ObjectDisposedException)
                    {
                        // Ignore if already disposed
                    }
                });
            }
        }

        /// <summary>
        /// Gets the current number of items in the queue.
        /// </summary>
        public int Count => _collection.Count;

        /// <summary>
        /// Gets whether the queue is marked complete and empty.
        /// </summary>
        public bool IsCompleted => _collection.IsCompleted && _collection.Count == 0;

        /// <summary>
        /// Gets the maximum capacity of the queue.
        /// </summary>
        public int Capacity => _collection.BoundedCapacity;

        /// <summary>
        /// Adds an item to the queue. Blocks if the queue is full until space becomes available.
        /// </summary>
        /// <param name="item">The item to add.</param>
        /// <exception cref="OperationCanceledException">Thrown when cancellation token is triggered.</exception>
        /// <exception cref="InvalidOperationException">Thrown when queue is marked complete.</exception>
        public void Enqueue(T item)
        {
            var stopwatch = _enableStatistics ? Stopwatch.StartNew() : null;

            try
            {
                _collection.Add(item);
                UpdateStatisticsOnEnqueue(stopwatch);
            }
            catch (OperationCanceledException)
            {
                UpdateStatisticsOnProducerWait(stopstopwatch);
                throw;
            }
            catch (InvalidOperationException)
            {
                throw;
            }
        }

        /// <summary>
        /// Tries to add an item to the queue with a timeout.
        /// </summary>
        /// <param name="item">The item to add.</param>
        /// <param name="timeoutMilliseconds">The timeout in milliseconds.</param>
        /// <returns>True if the item was added successfully, false if timeout elapsed.</returns>
        /// <exception cref="OperationCanceledException">Thrown when cancellation token is triggered.</exception>
        /// <exception cref="InvalidOperationException">Thrown when queue is marked complete.</exception>
        public bool TryEnqueue(T item, int timeoutMilliseconds)
        {
            var stopwatch = _enableStatistics ? Stopwatch.StartNew() : null;

            try
            {
                bool success = _collection.TryAdd(item, timeoutMilliseconds);
                if (success)
                {
                    UpdateStatisticsOnEnqueue(stopwatch);
                }
                else
                {
                    UpdateStatisticsOnProducerWait(stopstopwatch);
                }
                return success;
            }
            catch (OperationCanceledException)
            {
                UpdateStatisticsOnProducerWait(stopstopwatch);
                throw;
            }
            catch (InvalidOperationException)
            {
                throw;
            }
        }

        /// <summary>
        /// Removes and returns an item from the queue. Blocks if the queue is empty until an item is available.
        /// </summary>
        /// <returns>The next item in FIFO order.</returns>
        /// <exception cref="OperationCanceledException">Thrown when cancellation token is triggered.</exception>
        public T Dequeue()
        {
            var stopwatch = _enableStatistics ? Stopwatch.StartNew() : null;

            try
            {
                T item = _collection.Take();
                UpdateStatisticsOnDequeue(stopwatch);
                return item;
            }
            catch (OperationCanceledException)
            {
                UpdateStatisticsOnConsumerWait(stopstopwatch);
                throw;
            }
        }

        /// <summary>
        /// Tries to remove an item from the queue with a timeout.
        /// </summary>
        /// <param name="item">When this method returns, contains the item if found, otherwise default value.</param>
        /// <param name="timeoutMilliseconds">The timeout in milliseconds.</param>
        /// <returns>True if an item was removed successfully, false if timeout elapsed.</returns>
        /// <exception cref="OperationCanceledException">Thrown when cancellation token is triggered.</exception>
        public bool TryDequeue(out T item, int timeoutMilliseconds)
        {
            var stopwatch = _enableStatistics ? Stopwatch.StartNew() : null;

            try
            {
                bool success = _collection.TryTake(out item, timeoutMilliseconds);
                if (success)
                {
                    UpdateStatisticsOnDequeue(stopwatch);
                }
                else
                {
                    UpdateStatisticsOnConsumerWait(stopstopwatch);
                }
                return success;
            }
            catch (OperationCanceledException)
            {
                UpdateStatisticsOnConsumerWait(stopstopwatch);
                throw;
            }
        }

        /// <summary>
        /// Peeks at the next item without removing it from the queue.
        /// </summary>
        /// <param name="item">When this method returns, contains the next item if found, otherwise default value.</param>
        /// <returns>True if an item is available, false if queue is empty.</returns>
        public bool TryPeek(out T item)
        {
            if (_collection.Count > 0)
            {
                item = _collection.First();
                return true;
            }

            item = default(T)!;
            return false;
        }

        /// <summary>
        /// Signals that no more items will be added to the queue.
        /// Existing items can still be dequeued.
        /// </summary>
        public void CompleteAdding()
        {
            _collection.CompleteAdding();
        }

        /// <summary>
        /// Waits for all items to be dequeued from the queue.
        /// Blocks until all items are dequeued or returns immediately if already complete.
        /// </summary>
        public void WaitForCompletion()
        {
            while (!_collection.IsCompleted || _collection.Count > 0)
            {
                if (!_collection.TryTake(out _, 100))
                {
                    if (_collection.IsCompleted && _collection.Count == 0)
                    {
                        break;
                    }
                }
            }
        }

        /// <summary>
        /// Immediately shuts down the queue, cancelling all blocking operations.
        /// </summary>
        public void Shutdown()
        {
            try
            {
                _collection.CompleteAdding();
                _collection.CancelPendingConsumers();
                _collection.CancelPendingProducers();
            }
            catch (ObjectDisposedException)
            {
                // Ignore if already disposed
            }
        }

        /// <summary>
        /// Enqueues multiple items atomically. More efficient than individual enqueues.
        /// </summary>
        /// <param name="items">The items to enqueue.</param>
        /// <exception cref="OperationCanceledException">Thrown when cancellation token is triggered.</exception>
        /// <exception cref="InvalidOperationException">Thrown when queue is marked complete.</exception>
        public void EnqueueBatch(IEnumerable<T> items)
        {
            var stopwatch = _enableStatistics ? Stopwatch.StartNew() : null;
            int count = 0;

            try
            {
                foreach (var item in items)
                {
                    _collection.Add(item);
                    count++;
                }
                UpdateStatisticsOnBatchEnqueue(stopwatch, count);
            }
            catch (OperationCanceledException)
            {
                UpdateStatisticsOnProducerWait(stopstopwatch);
                throw;
            }
            catch (InvalidOperationException)
            {
                throw;
            }
        }

        /// <summary>
        /// Tries to dequeue multiple items. More efficient than individual dequeues.
        /// </summary>
        /// <param name="batchSize">The number of items to dequeue.</param>
        /// <param name="timeoutMilliseconds">The timeout in milliseconds.</param>
        /// <returns>Array with available items (may be fewer than batchSize if timeout).</returns>
        public T[] DequeueBatch(int batchSize, int timeoutMilliseconds)
        {
            var result = new List<T>();
            var stopwatch = _enableStatistics ? Stopwatch.StartNew() : null;
            var endTime = DateTime.UtcNow.AddMilliseconds(timeoutMilliseconds);

            try
            {
                while (result.Count < batchSize && DateTime.UtcNow < endTime)
                {
                    int remainingTimeout = (int)(endTime - DateTime.UtcNow).TotalMilliseconds;
                    if (remainingTimeout <= 0)
                    {
                        break;
                    }

                    if (_collection.TryTake(out T item, remainingTimeout))
                    {
                        result.Add(item);
                    }
                }

                UpdateStatisticsOnBatchDequeue(stopwatch, result.Count);
                return result.ToArray();
            }
            catch (OperationCanceledException)
            {
                UpdateStatisticsOnConsumerWait(stopstopwatch);
                return result.ToArray();
            }
        }

        /// <summary>
        /// Gets the current queue statistics.
        /// </summary>
        /// <returns>Queue statistics object.</returns>
        public QueueStatistics GetStatistics()
        {
            lock (_statisticsLock)
            {
                return new QueueStatistics
                {
                    TotalEnqueued = _statistics.TotalEnqueued,
                    TotalDequeued = _statistics.TotalDequeued,
                    AverageWaitTimeMs = _statistics.TotalWaitCount > 0
                        ? _statistics.TotalWaitTimeMs / _statistics.TotalWaitCount
                        : 0,
                    MaxQueueSize = _statistics.MaxQueueSize,
                    ProducerWaitCount = _statistics.ProducerWaitCount,
                    ConsumerWaitCount = _statistics.ConsumerWaitCount
                };
            }
        }

        /// <summary>
        /// Releases all resources used by the queue.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                _collection.Dispose();
            }
        }

        #region Statistics Update Methods

        private void UpdateStatisticsOnEnqueue(Stopwatch? stopwatch)
        {
            if (!_enableStatistics) return;

            lock (_statisticsLock)
            {
                _statistics.TotalEnqueued++;
                if (_collection.Count > _statistics.MaxQueueSize)
                {
                    _statistics.MaxQueueSize = _collection.Count;
                }
            }
        }

        private void UpdateStatisticsOnDequeue(Stopwatch? stopwatch)
        {
            if (!_enableStatistics) return;

            lock (_statisticsLock)
            {
                _statistics.TotalDequeued++;
            }
        }

        private void UpdateStatisticsOnBatchEnqueue(Stopwatch? stopwatch, int count)
        {
            if (!_enableStatistics) return;

            lock (_statisticsLock)
            {
                _statistics.TotalEnqueued += count;
                if (_collection.Count > _statistics.MaxQueueSize)
                {
                    _statistics.MaxQueueSize = _collection.Count;
                }
            }
        }

        private void UpdateStatisticsOnBatchDequeue(Stopwatch? stopwatch, int count)
        {
            if (!_enableStatistics) return;

            lock (_statisticsLock)
            {
                _statistics.TotalDequeued += count;
            }
        }

        private void UpdateStatisticsOnProducerWait(Stopwatch? stopwatch)
        {
            if (!_enableStatistics) return;

            lock (_statisticsLock)
            {
                _statistics.ProducerWaitCount++;
                if (stopwatch != null)
                {
                    _statistics.TotalWaitTimeMs += stopwatch.ElapsedMilliseconds;
                    _statistics.TotalWaitCount++;
                }
            }
        }

        private void UpdateStatisticsOnConsumerWait(Stopwatch? stopwatch)
        {
            if (!_enableStatistics) return;

            lock (_statisticsLock)
            {
                _statistics.ConsumerWaitCount++;
                if (stopwatch != null)
                {
                    _statistics.TotalWaitTimeMs += stopwatch.ElapsedMilliseconds;
                    _statistics.TotalWaitCount++;
                }
            }
        }

        #endregion
    }
}
