namespace MachineLearning.Checkpointing.Async;

using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;

/// <summary>
/// Manager for asynchronous checkpointing operations
/// </summary>
public class AsyncCheckpointManager : IDisposable
{
    private readonly IDistributedCoordinator _coordinator;
    private readonly ICheckpointStorage _storage;
    private readonly DistributedCheckpoint _checkpoint;
    private readonly CancellationTokenSource _shutdownTokenSource;
    private readonly BlockingCollection<CheckpointTask> _taskQueue;
    private readonly Task _workerTask;
    private readonly ILogger<AsyncCheckpointManager>? _logger;
    private readonly object _stateLock = new();
    private readonly Dictionary<string, CheckpointTask> _activeTasks = new();

    /// <summary>
    /// Gets the maximum queue size
    /// </summary>
    public int MaxQueueSize { get; }

    /// <summary>
    /// Gets the number of tasks currently in the queue
    /// </summary>
    public int QueueSize => _taskQueue.Count;

    /// <summary>
    /// Gets the number of active tasks (queued + running)
    /// </summary>
    public int ActiveTaskCount
    {
        get
        {
            lock (_stateLock)
            {
                return _activeTasks.Count;
            }
        }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="AsyncCheckpointManager"/> class.
    /// </summary>
    /// <param name="coordinator">The distributed coordinator.</param>
    /// <param name="checkpoint">The distributed checkpoint instance.</param>
    /// <param name="maxQueueSize">The maximum number of queued checkpoint tasks.</param>
    /// <param name="logger">The logger instance.</param>
    public AsyncCheckpointManager(
        IDistributedCoordinator coordinator,
        DistributedCheckpoint checkpoint,
        int maxQueueSize = 10,
        ILogger<AsyncCheckpointManager>? logger = null)
    {
        _coordinator = coordinator ?? throw new ArgumentNullException(nameof(coordinator));
        _checkpoint = checkpoint ?? throw new ArgumentNullException(nameof(checkpoint));
        _storage = checkpoint.GetStorage() ?? throw new InvalidOperationException("Checkpoint storage is not available");
        _logger = logger;
        _shutdownTokenSource = new CancellationTokenSource();
        _taskQueue = new BlockingCollection<CheckpointTask>(maxQueueSize);
        MaxQueueSize = maxQueueSize;

        // Start background worker
        _workerTask = Task.Run(ProcessQueueAsync);

        _logger?.LogInformation("Async checkpoint manager initialized (max queue size: {MaxQueueSize})", maxQueueSize);
    }

    /// <summary>
    /// Queue a checkpoint save operation
    /// </summary>
    /// <param name="model">The model state to save.</param>
    /// <param name="optimizer">The optimizer state to save.</param>
    /// <param name="options">The save options.</param>
    /// <returns>The checkpoint ID for tracking.</returns>
    /// <exception cref="CheckpointQueueFullException">Thrown when the queue is full.</exception>
    public string QueueSaveAsync(
        IStateful model,
        IStateful optimizer,
        SaveOptions? options = null)
    {
        var checkpointId = Guid.NewGuid().ToString("N");
        var task = new CheckpointTask
        {
            Id = checkpointId,
            Type = CheckpointTaskType.Save,
            Model = model,
            Optimizer = optimizer,
            Options = options ?? new SaveOptions(),
            QueuedAt = DateTime.UtcNow,
            Status = CheckpointTaskStatus.Queued
        };

        lock (_stateLock)
        {
            _activeTasks[checkpointId] = task;
        }

        try
        {
            _taskQueue.Add(task);
            _logger?.LogInformation("Queued checkpoint save: {CheckpointId}", checkpointId);
        }
        catch (InvalidOperationException)
        {
            // Queue is full or complete
            lock (_stateLock)
            {
                task.Status = CheckpointTaskStatus.Rejected;
                _activeTasks.Remove(checkpointId);
            }
            _logger?.LogWarning("Checkpoint queue full, rejecting: {CheckpointId}", checkpointId);
            throw new CheckpointQueueFullException(
                $"Checkpoint queue is full (current: {_taskQueue.Count}, max: {MaxQueueSize})",
                checkpointId,
                _taskQueue.Count,
                MaxQueueSize);
        }

        return checkpointId;
    }

    /// <summary>
    /// Get the status of a checkpoint task
    /// </summary>
    /// <param name="checkpointId">The checkpoint ID.</param>
    /// <returns>The task status, or null if not found.</returns>
    public CheckpointTaskStatus? GetTaskStatus(string checkpointId)
    {
        lock (_stateLock)
        {
            if (_activeTasks.TryGetValue(checkpointId, out var task))
            {
                return task.Status;
            }
            return null;
        }
    }

    /// <summary>
    /// Wait for a checkpoint to complete
    /// </summary>
    /// <param name="checkpointId">The checkpoint ID.</param>
    /// <param name="timeout">Optional timeout duration.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The checkpoint task result.</returns>
    /// <exception cref="ArgumentException">Thrown when checkpoint ID is not found.</exception>
    /// <exception cref="TimeoutException">Thrown when the checkpoint does not complete within the timeout.</exception>
    public async Task<CheckpointTaskResult> WaitForCompletionAsync(
        string checkpointId,
        TimeSpan? timeout = null,
        CancellationToken cancellationToken = default)
    {
        CheckpointTask? task;
        lock (_stateLock)
        {
            _activeTasks.TryGetValue(checkpointId, out task);
        }

        if (task == null)
        {
            throw new ArgumentException($"Checkpoint task not found: {checkpointId}", nameof(checkpointId));
        }

        var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        if (timeout.HasValue)
        {
            cts.CancelAfter(timeout.Value);
        }

        try
        {
            await Task.Delay(Timeout.InfiniteTimeSpan, cts.Token)
                .ContinueWith(_ => { }, cancellationToken, TaskContinuationOptions.None, TaskScheduler.Default);
        }
        catch (OperationCanceledException) when (cts.Token.IsCancellationRequested)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            throw new TimeoutException($"Checkpoint {checkpointId} did not complete within timeout");
        }

        return task.Result ?? throw new InvalidOperationException($"Checkpoint {checkpointId} completed without result");
    }

    /// <summary>
    /// Cancel a pending checkpoint
    /// </summary>
    /// <param name="checkpointId">The checkpoint ID.</param>
    /// <returns>True if the checkpoint was cancelled, false otherwise.</returns>
    public bool CancelCheckpoint(string checkpointId)
    {
        CheckpointTask? task;
        lock (_stateLock)
        {
            if (_activeTasks.TryGetValue(checkpointId, out task))
            {
                if (task.Status == CheckpointTaskStatus.Queued)
                {
                    task.Status = CheckpointTaskStatus.Cancelled;
                    task.CompletionSource.SetCanceled();
                    _logger?.LogInformation("Cancelled checkpoint: {CheckpointId}", checkpointId);
                    return true;
                }
            }
        }
        return false;
    }

    /// <summary>
    /// Get all active tasks
    /// </summary>
    /// <returns>List of active checkpoint tasks.</returns>
    public List<CheckpointTaskInfo> GetActiveTasks()
    {
        lock (_stateLock)
        {
            return _activeTasks.Values.Select(t => new CheckpointTaskInfo
            {
                Id = t.Id,
                Type = t.Type,
                Status = t.Status,
                QueuedAt = t.QueuedAt,
                StartedAt = t.StartedAt,
                CompletedAt = t.CompletedAt
            }).ToList();
        }
    }

    /// <summary>
    /// Get task info by ID
    /// </summary>
    /// <param name="checkpointId">The checkpoint ID.</param>
    /// <returns>Task info, or null if not found.</returns>
    public CheckpointTaskInfo? GetTaskInfo(string checkpointId)
    {
        lock (_stateLock)
        {
            if (_activeTasks.TryGetValue(checkpointId, out var task))
            {
                return new CheckpointTaskInfo
                {
                    Id = task.Id,
                    Type = task.Type,
                    Status = task.Status,
                    QueuedAt = task.QueuedAt,
                    StartedAt = task.StartedAt,
                    CompletedAt = task.CompletedAt
                };
            }
            return null;
        }
    }

    /// <summary>
    /// Process the checkpoint queue in the background
    /// </summary>
    private async Task ProcessQueueAsync()
    {
        _logger?.LogInformation("Background checkpoint worker started");

        try
        {
            while (!_shutdownTokenSource.Token.IsCancellationRequested)
            {
                CheckpointTask task;
                try
                {
                    task = _taskQueue.Take(_shutdownTokenSource.Token);
                }
                catch (InvalidOperationException)
                {
                    // Queue is complete
                    break;
                }
                catch (OperationCanceledException)
                {
                    // Shutdown requested
                    break;
                }

                await ProcessTaskAsync(task);
            }
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Background checkpoint worker encountered an error");
        }
        finally
        {
            _logger?.LogInformation("Background checkpoint worker stopped");
        }
    }

    /// <summary>
    /// Process a single checkpoint task
    /// </summary>
    private async Task ProcessTaskAsync(CheckpointTask task)
    {
        // Check if task was cancelled while queued
        lock (_stateLock)
        {
            if (task.Status == CheckpointTaskStatus.Cancelled)
            {
                return;
            }

            task.Status = CheckpointTaskStatus.Running;
            task.StartedAt = DateTime.UtcNow;
        }

        _logger?.LogInformation("Processing checkpoint: {CheckpointId}", task.Id);

        try
        {
            string checkpointPath;
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            switch (task.Type)
            {
                case CheckpointTaskType.Save:
                    if (task.Model == null || task.Optimizer == null)
                    {
                        throw new InvalidOperationException("Model and optimizer must not be null for save operation");
                    }
                    checkpointPath = await _checkpoint.SaveAsync(
                        task.Model,
                        task.Optimizer,
                        task.Options,
                        _shutdownTokenSource.Token);
                    break;

                case CheckpointTaskType.Load:
                    throw new NotImplementedException("Async load not implemented");

                default:
                    throw new ArgumentException($"Unknown task type: {task.Type}");
            }

            stopwatch.Stop();

            // Update task status to completed
            lock (_stateLock)
            {
                task.Status = CheckpointTaskStatus.Completed;
                task.CompletedAt = DateTime.UtcNow;
                task.Result = CheckpointTaskResult.CreateSuccess(checkpointPath, stopwatch.Elapsed);
            }

            task.CompletionSource.SetResult(task.Result);

            _logger?.LogInformation("Checkpoint completed: {CheckpointId} -> {CheckpointPath} (Duration: {Duration}ms)",
                task.Id, checkpointPath, stopwatch.ElapsedMilliseconds);
        }
        catch (Exception ex)
        {
            // Update task status to failed
            lock (_stateLock)
            {
                task.Status = CheckpointTaskStatus.Failed;
                task.CompletedAt = DateTime.UtcNow;
                task.Result = CheckpointTaskResult.CreateFailure(ex.Message, ex);
            }

            task.CompletionSource.SetResult(task.Result);

            _logger?.LogError(ex, "Checkpoint failed: {CheckpointId}", task.Id);
        }
        finally
        {
            // Clean up old completed tasks from active tasks dictionary
            // Keep only recent completed tasks (completed within last 10 minutes)
            lock (_stateLock)
            {
                var cutoff = DateTime.UtcNow.AddMinutes(-10);
                var oldTasks = _activeTasks
                    .Where(kvp => kvp.Value.CompletedAt.HasValue && kvp.Value.CompletedAt.Value < cutoff)
                    .Select(kvp => kvp.Key)
                    .ToList();

                foreach (var oldTaskId in oldTasks)
                {
                    _activeTasks.Remove(oldTaskId);
                }
            }
        }
    }

    /// <summary>
    /// Dispose resources
    /// </summary>
    public void Dispose()
    {
        _logger?.LogInformation("Shutting down async checkpoint manager");

        _shutdownTokenSource.Cancel();
        _taskQueue.CompleteAdding();

        try
        {
            if (!_workerTask.Wait(TimeSpan.FromSeconds(30)))
            {
                _logger?.LogWarning("Background worker did not shut down gracefully within timeout");
            }
        }
        catch (AggregateException ex)
        {
            _logger?.LogError(ex, "Error during shutdown");
        }

        _shutdownTokenSource.Dispose();
        _taskQueue.Dispose();

        _logger?.LogInformation("Async checkpoint manager shutdown complete");
    }
}
