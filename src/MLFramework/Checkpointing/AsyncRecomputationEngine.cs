namespace MLFramework.Checkpointing;

/// <summary>
/// Asynchronous recomputation engine for parallel execution
/// </summary>
public class AsyncRecomputationEngine : IDisposable
{
    private readonly Dictionary<string, Func<Tensor>> _recomputeFunctions;
    private readonly SemaphoreSlim _semaphore;
    private readonly object _lock = new object();
    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of AsyncRecomputationEngine
    /// </summary>
    /// <param name="maxDegreeOfParallelism">Maximum parallel operations</param>
    public AsyncRecomputationEngine(int maxDegreeOfParallelism = -1)
    {
        _recomputeFunctions = new Dictionary<string, Func<Tensor>>();
        _semaphore = new SemaphoreSlim(maxDegreeOfParallelism <= 0
            ? Environment.ProcessorCount
            : maxDegreeOfParallelism);
        _disposed = false;
    }

    /// <summary>
    /// Registers a recompute function
    /// </summary>
    public void RegisterRecomputeFunction(string layerId, Func<Tensor> recomputeFunction)
    {
        if (string.IsNullOrWhiteSpace(layerId))
            throw new ArgumentException("Layer ID cannot be null or whitespace");
        if (recomputeFunction == null)
            throw new ArgumentNullException(nameof(recomputeFunction));

        lock (_lock)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(AsyncRecomputationEngine));

            if (_recomputeFunctions.ContainsKey(layerId))
            {
                throw new ArgumentException($"Layer '{layerId}' is already registered");
            }

            _recomputeFunctions[layerId] = recomputeFunction;
        }
    }

    /// <summary>
    /// Asynchronously recomputes an activation
    /// </summary>
    public async Task<Tensor> RecomputeAsync(string layerId, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(layerId))
            throw new ArgumentException("Layer ID cannot be null or whitespace");

        Func<Tensor>? recomputeFunc;

        lock (_lock)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(AsyncRecomputationEngine));

            if (!_recomputeFunctions.TryGetValue(layerId, out recomputeFunc))
            {
                throw new KeyNotFoundException($"No recompute function registered for layer '{layerId}'");
            }
        }

        await _semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);

        try
        {
            return await Task.Run(() => recomputeFunc(), cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _semaphore.Release();
        }
    }

    /// <summary>
    /// Asynchronously recomputes multiple activations in parallel
    /// </summary>
    public async Task<Dictionary<string, Tensor>> RecomputeMultipleAsync(
        IEnumerable<string> layerIds,
        CancellationToken cancellationToken = default)
    {
        if (layerIds == null)
            throw new ArgumentNullException(nameof(layerIds));

        var layerList = layerIds.ToList();
        if (layerList.Count == 0)
            return new Dictionary<string, Tensor>();

        var tasks = layerList.Select(async layerId =>
        {
            var result = await RecomputeAsync(layerId, cancellationToken).ConfigureAwait(false);
            return (LayerId: layerId, Tensor: result);
        });

        var results = await Task.WhenAll(tasks).ConfigureAwait(false);

        return results.ToDictionary(r => r.LayerId, r => r.Tensor);
    }

    /// <summary>
    /// Disposes the engine and releases resources
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            lock (_lock)
            {
                _recomputeFunctions.Clear();
            }
            _semaphore.Dispose();
            _disposed = true;
        }
    }
}
