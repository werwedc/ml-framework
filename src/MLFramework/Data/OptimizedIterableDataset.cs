using System;
using System.Collections;
using System.Collections.Generic;

namespace MLFramework.Data;

/// <summary>
/// Optimized version of IterableDataset with support for multi-process workers,
/// iterator caching, and stream replication.
/// </summary>
/// <typeparam name="T">The type of data items in the dataset.</typeparam>
public class OptimizedIterableDataset<T> : IterableDataset<T>
{
    private readonly Func<IEnumerator<T>> _iteratorFactory;
    private IEnumerator<T>? _cachedIterator;
    private readonly bool _enableWorkerSupport;
    private readonly int _workerId;
    private readonly int _totalWorkers;
    private volatile bool _iteratorCreated;
    private readonly object _lock = new object();

    /// <summary>
    /// Initializes a new instance of the OptimizedIterableDataset class.
    /// </summary>
    /// <param name="iteratorFactory">Factory function to create new iterators.</param>
    /// <param name="enableWorkerSupport">Enable multi-process worker support.</param>
    /// <param name="workerId">The ID of this worker (0-based).</param>
    /// <param name="totalWorkers">Total number of workers.</param>
    public OptimizedIterableDataset(
        Func<IEnumerator<T>> iteratorFactory,
        bool enableWorkerSupport = false,
        int workerId = 0,
        int totalWorkers = 1)
    {
        _iteratorFactory = iteratorFactory ?? throw new ArgumentNullException(nameof(iteratorFactory));
        _enableWorkerSupport = enableWorkerSupport;
        _workerId = workerId;
        _totalWorkers = totalWorkers;
        _cachedIterator = null;
        _iteratorCreated = false;

        ValidateWorkerParameters(workerId, totalWorkers);
    }

    /// <summary>
    /// Returns an enumerator that iterates through the dataset.
    /// </summary>
    /// <returns>An enumerator that can be used to iterate through the dataset.</returns>
    public override IEnumerator<T> GetEnumerator()
    {
        lock (_lock)
        {
            if (!_iteratorCreated)
            {
                _cachedIterator = CreateIterator();
                _iteratorCreated = true;
            }
        }

        return CreateStreamIterator();
    }

    /// <summary>
    /// Resets the dataset, clearing any cached iterator.
    /// Should be called before starting a new epoch.
    /// </summary>
    public void Reset()
    {
        lock (_lock)
        {
            _cachedIterator?.Dispose();
            _cachedIterator = null;
            _iteratorCreated = false;
        }
    }

    /// <summary>
    /// Creates the appropriate iterator based on configuration.
    /// </summary>
    private IEnumerator<T> CreateIterator()
    {
        if (_enableWorkerSupport && _totalWorkers > 1)
        {
            // Create worker-aware iterator
            return CreateWorkerIterator();
        }

        return _iteratorFactory();
    }

    /// <summary>
    /// Creates a worker-aware iterator that skips to the worker's position in the stream.
    /// Each worker processes samples with stride equal to totalWorkers.
    /// </summary>
    private IEnumerator<T> CreateWorkerIterator()
    {
        var baseIterator = _iteratorFactory();
        var sampleCount = 0;

        // Process samples with stride-based partitioning
        while (baseIterator.MoveNext())
        {
            // Each worker processes samples where (sampleCount % totalWorkers == workerId)
            if (sampleCount % _totalWorkers == _workerId)
            {
                yield return baseIterator.Current;
            }
            sampleCount++;
        }

        baseIterator.Dispose();
    }

    /// <summary>
    /// Creates a stream iterator that wraps the cached iterator.
    /// </summary>
    private IEnumerator<T> CreateStreamIterator()
    {
        if (_cachedIterator == null)
        {
            throw new InvalidOperationException("Iterator not created. Call GetEnumerator first.");
        }

        return _cachedIterator;
    }

    /// <summary>
    /// Validates worker parameters.
    /// </summary>
    private static void ValidateWorkerParameters(int workerId, int totalWorkers)
    {
        if (workerId < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(workerId), "Worker ID cannot be negative.");
        }

        if (totalWorkers < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(totalWorkers), "Total workers must be at least 1.");
        }

        if (workerId >= totalWorkers)
        {
            throw new ArgumentOutOfRangeException(
                nameof(workerId),
                $"Worker ID ({workerId}) must be less than total workers ({totalWorkers}).");
        }
    }
}
