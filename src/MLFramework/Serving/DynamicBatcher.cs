using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace MLFramework.Serving;

/// <summary>
/// Delegate for processing a batch of requests
/// </summary>
/// <param name="requests">List of request payloads</param>
/// <returns>List of responses corresponding to input requests</returns>
public delegate Task<List<TResponse>> BatchProcessor<TRequest, TResponse>(
    List<TRequest> requests);

/// <summary>
/// Dynamic batching system that groups incoming requests for efficient processing
/// </summary>
public class DynamicBatcher<TRequest, TResponse> : IDisposable
{
    private readonly BatchingConfiguration _config;
    private readonly BoundedRequestQueue<TRequest, TResponse> _queue;
    private readonly BatchBuilder<TRequest, TResponse> _batchBuilder;
    private readonly BatchProcessor<TRequest, TResponse> _batchProcessor;
    private readonly ResponseScatterer<TResponse> _responseScatterer;
    private readonly CancellationTokenSource _shutdownCts;
    private readonly Task _processingTask;
    private bool _disposed;

    /// <summary>
    /// Create a new dynamic batcher
    /// </summary>
    /// <param name="config">Batching configuration</param>
    /// <param name="batchProcessor">Delegate for processing batches</param>
    public DynamicBatcher(
        BatchingConfiguration config,
        BatchProcessor<TRequest, TResponse> batchProcessor)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _config.Validate();
        _batchProcessor = batchProcessor ?? throw new ArgumentNullException(nameof(batchProcessor));

        _queue = new BoundedRequestQueue<TRequest, TResponse>(config.MaxQueueSize);
        _batchBuilder = new BatchBuilder<TRequest, TResponse>(_queue, config);
        _responseScatterer = new ResponseScatterer<TResponse>();
        _shutdownCts = new CancellationTokenSource();

        // Start background processing task
        _processingTask = Task.Run(() => ProcessingLoop(_shutdownCts.Token));
    }

    /// <summary>
    /// Submit a request for batched processing
    /// </summary>
    /// <param name="request">The request to process</param>
    /// <param name="cancellationToken">Optional cancellation token</param>
    /// <returns>Response for the request</returns>
    public async Task<TResponse> ProcessAsync(
        TRequest request,
        CancellationToken cancellationToken = default)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(DynamicBatcher<TRequest, TResponse>));

        // Create queued request with task completion source
        var queuedRequest = new QueuedRequest<TRequest, TResponse>(request);

        // Try to enqueue
        var enqueued = await _queue.TryEnqueueAsync(queuedRequest, cancellationToken);
        if (!enqueued)
            throw new InvalidOperationException("Request queue is full. Please retry later.");

        // Return task that will be completed when response is ready
        return await queuedRequest.ResponseSource.Task;
    }

    private async Task ProcessingLoop(CancellationToken cancellationToken)
    {
        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    // Build a batch
                    var batch = await _batchBuilder.BuildBatchAsync(cancellationToken);

                    if (batch.Requests.Count == 0)
                        continue;

                    // Process batch asynchronously
                    _ = Task.Run(() => ProcessBatch(batch), cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    // Expected during shutdown
                    break;
                }
                catch (Exception ex)
                {
                    // Log error but continue processing
                    Console.Error.WriteLine($"Error in batch processing loop: {ex.Message}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Fatal error in processing loop: {ex.Message}");
        }
    }

    private async Task ProcessBatch(RequestBatch<TRequest, TResponse> batch)
    {
        List<TResponse> responses = null;
        Exception processingException = null;

        try
        {
            // Get request payloads
            var payloads = batch.GetPayloads();

            // Process batch
            responses = await _batchProcessor(payloads);
        }
        catch (Exception ex)
        {
            processingException = ex;
        }

        // Scatter responses back to individual requests
        _responseScatterer.Scatter(batch.Requests, responses, processingException);
    }

    /// <summary>
    /// Get current queue statistics
    /// </summary>
    public QueueStatistics GetStatistics()
    {
        return new QueueStatistics
        {
            CurrentQueueSize = _queue.Count,
            IsQueueFull = _queue.IsFull,
            IsQueueEmpty = _queue.IsEmpty
        };
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;

        // Signal shutdown
        _shutdownCts.Cancel();

        // Wait for processing task to complete (with timeout)
        try
        {
            _processingTask.Wait(TimeSpan.FromSeconds(5));
        }
        catch (AggregateException)
        {
            // Task was cancelled or failed
        }

        _shutdownCts.Dispose();

        // Cancel any pending requests
        var remaining = _queue.Dequeue(_queue.Count);
        foreach (var request in remaining)
        {
            request.ResponseSource.TrySetCanceled();
        }
    }
}

/// <summary>
/// Current state of the request queue
/// </summary>
public class QueueStatistics
{
    public int CurrentQueueSize { get; set; }
    public bool IsQueueFull { get; set; }
    public bool IsQueueEmpty { get; set; }
}
