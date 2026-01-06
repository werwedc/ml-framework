using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace MLFramework.Data.Worker
{
    /// <summary>
    /// Prefetch queue that proactively prepares data batches before they're needed.
    /// </summary>
    /// <typeparam name="T">The type of items being prefetched.</typeparam>
    public class PrefetchQueue<T> : IPrefetchQueue<T>
    {
        private readonly int _prefetchCount;
        private readonly SharedMemoryQueue<T> _queue;
        private readonly CancellationTokenSource _cancellationToken;
        private Task _prefetchTask;
        private readonly Func<IEnumerable<T>> _batchGenerator;
        private volatile bool _isRunning;

        /// <summary>
        /// Initializes a new instance of the PrefetchQueue class.
        /// </summary>
        /// <param name="batchGenerator">Function that generates batches to be prefetched.</param>
        /// <param name="prefetchCount">Number of batches to prepare ahead (default: 2).</param>
        public PrefetchQueue(
            Func<IEnumerable<T>> batchGenerator,
            int prefetchCount = 2)
        {
            if (batchGenerator == null)
                throw new ArgumentNullException(nameof(batchGenerator));

            if (prefetchCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(prefetchCount), "prefetchCount must be positive");

            _batchGenerator = batchGenerator;
            _prefetchCount = prefetchCount;
            _cancellationToken = new CancellationTokenSource();
            _queue = new SharedMemoryQueue<T>(maxSize: prefetchCount * 2);
            _isRunning = false;
            _prefetchTask = null;
        }

        /// <inheritdoc/>
        public void Start()
        {
            if (_isRunning)
                return;

            _isRunning = true;

            _prefetchTask = Task.Run(() => PrefetchLoop(_cancellationToken.Token));
        }

        /// <summary>
        /// Background loop that continuously refills the prefetch buffer.
        /// </summary>
        private void PrefetchLoop(CancellationToken token)
        {
            while (!token.IsCancellationRequested)
            {
                try
                {
                    // Check if we need more batches
                    while (_queue.Count < _prefetchCount && !token.IsCancellationRequested)
                    {
                        try
                        {
                            // Generate next batch
                            var batchIterator = _batchGenerator();

                            if (batchIterator == null)
                                break;

                            foreach (var batch in batchIterator)
                            {
                                _queue.Enqueue(batch);

                                // Check cancellation after each batch
                                if (token.IsCancellationRequested)
                                    break;
                            }

                            // Small delay to prevent busy-waiting if generator returned empty
                            if (_queue.Count == 0)
                                break;
                        }
                        catch (Exception ex)
                        {
                            // Log error but continue
                            Console.WriteLine($"Prefetch error: {ex.Message}");
                            break;
                        }
                    }

                    // Small sleep to prevent busy-waiting
                    if (!token.IsCancellationRequested)
                        Thread.Sleep(1);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    // Log unexpected errors
                    Console.WriteLine($"Prefetch loop error: {ex.Message}");
                    Thread.Sleep(10);
                }
            }
        }

        /// <inheritdoc/>
        public T GetNext()
        {
            if (!_isRunning)
                throw new InvalidOperationException("Prefetch queue is not running");

            return _queue.Dequeue();
        }

        /// <inheritdoc/>
        public bool TryGetNext(out T batch)
        {
            if (!_isRunning)
            {
                batch = default;
                return false;
            }

            return _queue.TryDequeue(out batch);
        }

        /// <inheritdoc/>
        public bool IsRunning => _isRunning;

        /// <inheritdoc/>
        public int PrefetchCount => _prefetchCount;

        /// <inheritdoc/>
        public int AvailableBatches => _queue.Count;

        /// <inheritdoc/>
        public void Stop()
        {
            if (!_isRunning)
                return;

            _cancellationToken.Cancel();

            try
            {
                _prefetchTask?.Wait(TimeSpan.FromSeconds(5));
            }
            catch (AggregateException)
            {
                // Ignore task cancellation exceptions
            }

            _isRunning = false;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            Stop();
            _cancellationToken?.Dispose();
            _queue?.Dispose();
            _prefetchTask?.Dispose();
        }
    }
}
