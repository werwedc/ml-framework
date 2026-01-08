# Spec: Async Checkpointing (Background Saves)

## Overview
Implement asynchronous checkpointing to overlap checkpoint I/O with training, minimizing training interruption. Includes background thread management, queueing, and progress tracking.

## Scope
- 30-45 minutes coding time
- Focus on async operations and threading
- Target: `src/MLFramework/Checkpointing/Async/`

## Classes

### 1. AsyncCheckpointManager (Main Manager)
```csharp
public class AsyncCheckpointManager : IDisposable
{
    private readonly IDistributedCoordinator _coordinator;
    private readonly ICheckpointStorage _storage;
    private readonly IFaultToleranceHandler _faultHandler;
    private readonly DistributedCheckpoint _checkpoint;
    private readonly CancellationTokenSource _shutdownTokenSource;
    private readonly BlockingCollection<CheckpointTask> _taskQueue;
    private readonly Task _workerTask;
    private readonly ILogger<AsyncCheckpointManager> _logger;
    private readonly object _stateLock = new();
    private Dictionary<string, CheckpointTask> _activeTasks = new();

    public AsyncCheckpointManager(
        IDistributedCoordinator coordinator,
        DistributedCheckpoint checkpoint,
        int maxQueueSize = 10,
        ILogger<AsyncCheckpointManager>? logger = null)
    {
        _coordinator = coordinator;
        _checkpoint = checkpoint;
        _storage = checkpoint.GetStorage(); // Need to add this getter
        _faultHandler = checkpoint.GetFaultHandler(); // Need to add this getter
        _logger = logger;
        _shutdownTokenSource = new CancellationTokenSource();
        _taskQueue = new BlockingCollection<CheckpointTask>(maxQueueSize);

        // Start background worker
        _workerTask = Task.Run(ProcessQueueAsync);

        _logger?.LogInformation("Async checkpoint manager initialized (max queue size: {MaxQueueSize})", maxQueueSize);
    }

    /// <summary>
    /// Queue a checkpoint save operation
    /// </summary>
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
            }
            _logger?.LogWarning("Checkpoint queue full, rejecting: {CheckpointId}", checkpointId);
            throw new CheckpointQueueFullException("Checkpoint queue is full");
        }

        return checkpointId;
    }

    /// <summary>
    /// Get the status of a checkpoint task
    /// </summary>
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
            throw new ArgumentException($"Checkpoint task not found: {checkpointId}");
        }

        var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        if (timeout.HasValue)
        {
            cts.CancelAfter(timeout.Value);
        }

        while (!task.CompletionTask.IsCompleted && !cts.Token.IsCancellationRequested)
        {
            await Task.Delay(100, cts.Token);
        }

        if (cts.Token.IsCancellationRequested)
        {
            throw new TimeoutException($"Checkpoint {checkpointId} did not complete within timeout");
        }

        return task.Result ?? throw new InvalidOperationException($"Checkpoint {checkpointId} completed without result");
    }

    /// <summary>
    /// Cancel a pending checkpoint
    /// </summary>
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
                    return true;
                }
            }
        }
        return false;
    }

    /// <summary>
    /// Get all active tasks
    /// </summary>
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

    private async Task ProcessQueueAsync()
    {
        _logger?.LogInformation("Background checkpoint worker started");

        try
        {
            await foreach (var task in _taskQueue.GetConsumingAsyncEnumerable(_shutdownTokenSource.Token))
            {
                await ProcessTaskAsync(task);
            }
        }
        catch (OperationCanceledException)
        {
            // Shutdown requested
        }
        finally
        {
            _logger?.LogInformation("Background checkpoint worker stopped");
        }
    }

    private async Task ProcessTaskAsync(CheckpointTask task)
    {
        // Update task status
        lock (_stateLock)
        {
            task.Status = CheckpointTaskStatus.Running;
            task.StartedAt = DateTime.UtcNow;
        }

        _logger?.LogInformation("Processing checkpoint: {CheckpointId}", task.Id);

        try
        {
            string checkpointPath;
            switch (task.Type)
            {
                case CheckpointTaskType.Save:
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

            // Update task status to completed
            lock (_stateLock)
            {
                task.Status = CheckpointTaskStatus.Completed;
                task.CompletedAt = DateTime.UtcNow;
                task.Result = new CheckpointTaskResult
                {
                    Success = true,
                    CheckpointPath = checkpointPath
                };
            }

            _logger?.LogInformation("Checkpoint completed: {CheckpointId} -> {CheckpointPath}", task.Id, checkpointPath);
        }
        catch (Exception ex)
        {
            // Update task status to failed
            lock (_stateLock)
            {
                task.Status = CheckpointTaskStatus.Failed;
                task.CompletedAt = DateTime.UtcNow;
                task.Result = new CheckpointTaskResult
                {
                    Success = false,
                    Error = ex.Message
                };
            }

            _logger?.LogError(ex, "Checkpoint failed: {CheckpointId}", task.Id);
        }
    }

    public void Dispose()
    {
        _logger?.LogInformation("Shutting down async checkpoint manager");

        _shutdownTokenSource.Cancel();
        _taskQueue.CompleteAdding();

        try
        {
            _workerTask.Wait(TimeSpan.FromSeconds(30));
        }
        catch (AggregateException ex)
        {
            _logger?.LogError(ex, "Error during shutdown");
        }

        _shutdownTokenSource.Dispose();
        _taskQueue.Dispose();
    }
}
```

### 2. CheckpointTask (Task Representation)
```csharp
public class CheckpointTask
{
    public string Id { get; set; }
    public CheckpointTaskType Type { get; set; }
    public IStateful Model { get; set; }
    public IStateful Optimizer { get; set; }
    public SaveOptions Options { get; set; }
    public DateTime QueuedAt { get; set; }
    public DateTime? StartedAt { get; set; }
    public DateTime? CompletedAt { get; set; }
    public CheckpointTaskStatus Status { get; set; }
    public TaskCompletionSource<CheckpointTaskResult> CompletionSource { get; set; } = new();
    public Task<CheckpointTaskResult> CompletionTask => CompletionSource.Task;
    public CheckpointTaskResult? Result { get; set; }
}

public enum CheckpointTaskType
{
    Save,
    Load
}

public enum CheckpointTaskStatus
{
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
    Rejected
}
```

### 3. CheckpointTaskResult (Task Result)
```csharp
public class CheckpointTaskResult
{
    public bool Success { get; set; }
    public string? CheckpointPath { get; set; }
    public string? Error { get; set; }
    public TimeSpan Duration { get; set; }
    public long BytesWritten { get; set; }
}
```

### 4. CheckpointTaskInfo (Task Information)
```csharp
public class CheckpointTaskInfo
{
    public string Id { get; set; }
    public CheckpointTaskType Type { get; set; }
    public CheckpointTaskStatus Status { get; set; }
    public DateTime QueuedAt { get; set; }
    public DateTime? StartedAt { get; set; }
    public DateTime? CompletedAt { get; set; }

    public TimeSpan? Duration => CompletedAt - StartedAt;
}
```

### 5. AsyncCheckpointExtension (Extension Methods)
```csharp
public static class AsyncCheckpointExtension
{
    public static AsyncCheckpointManager CreateAsyncManager(
        this DistributedCheckpoint checkpoint,
        int maxQueueSize = 10)
    {
        // Need to access coordinator and storage
        // For now, assume these are accessible
        var coordinator = checkpoint.GetCoordinator(); // Need to add this getter
        return new AsyncCheckpointManager(coordinator, checkpoint, maxQueueSize);
    }
}
```

### 6. CheckpointQueueFullException (Exception)
```csharp
public class CheckpointQueueFullException : CheckpointException
{
    public CheckpointQueueFullException(string message)
        : base(message, ExceptionType.QueueFull)
    {
    }
}
```

## Usage Examples

### Basic Async Save
```csharp
var checkpoint = DistributedCheckpointFactory.Create(coordinator);
var asyncManager = checkpoint.CreateAsyncManager(maxQueueSize: 5);

// Queue checkpoint save (non-blocking)
var checkpointId = asyncManager.QueueSaveAsync(model, optimizer, new SaveOptions
{
    CheckpointPrefix = "async_model"
});

// Continue training...

// Wait for checkpoint to complete later
var result = await asyncManager.WaitForCompletionAsync(checkpointId);
Console.WriteLine($"Checkpoint saved to: {result.CheckpointPath}");
```

### With Timeout
```csharp
var result = await asyncManager.WaitForCompletionAsync(
    checkpointId,
    timeout: TimeSpan.FromMinutes(10));
```

### Monitor Task Status
```csharp
var status = asyncManager.GetTaskStatus(checkpointId);
Console.WriteLine($"Checkpoint status: {status}");

if (status == CheckpointTaskStatus.Completed)
{
    var result = await asyncManager.WaitForCompletionAsync(checkpointId);
}
```

### Cancel Pending Checkpoint
```csharp
var cancelled = asyncManager.CancelCheckpoint(checkpointId);
if (cancelled)
{
    Console.WriteLine("Checkpoint cancelled");
}
```

### List Active Tasks
```csharp
var activeTasks = asyncManager.GetActiveTasks();
foreach (var task in activeTasks)
{
    Console.WriteLine($"{task.Id}: {task.Status} (queued: {task.QueuedAt})");
}
```

## Integration Points
- Used by: Training loops for background checkpointing
- Depends on: `DistributedCheckpoint`, `IDistributedCoordinator`, `ICheckpointStorage`

## Thread Safety
- Task queue is thread-safe (BlockingCollection)
- Active tasks dictionary is protected by lock
- Each task has its own completion source

## Cleanup
- Properly disposes of cancellation tokens
- Waits for worker to complete during shutdown
- Cleans up task queue

## Testing Requirements
- Test queuing and processing
- Test concurrent queuing from multiple threads
- Test queue full scenario
- Test cancellation
- Test timeout handling
- Test status queries
- Test cleanup on dispose
- Test error handling in background task

## Success Criteria
- Checkpoint saves happen in background without blocking
- Queue prevents too many concurrent saves
- Can track status of queued checkpoints
- Can wait for completion with timeout
- Clean shutdown and resource cleanup
- Thread-safe operations
- Comprehensive error handling
