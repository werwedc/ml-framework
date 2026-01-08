# Spec: Error Handling and Robustness

## Overview
Implement comprehensive error handling, worker crash detection, timeout management, and graceful recovery mechanisms for the data loading pipeline.

## Requirements

### 1. WorkerError Class
Encapsulates information about worker errors.

```csharp
public class WorkerError
{
    public int WorkerId { get; }
    public Exception Exception { get; }
    public DateTime Timestamp { get; }
    public string Context { get; }

    public WorkerError(int workerId, Exception exception, string context = "")
}
```

### 2. ErrorPolicy Enum
Defines how errors should be handled.

```csharp
public enum ErrorPolicy
{
    FailFast,           // Stop entire dataloader on any error
    Continue,           // Skip failed worker, continue with others
    Restart,            // Attempt to restart failed worker
    Ignore              // Silently ignore errors (not recommended)
}
```

### 3. Error Handling Configuration

**Add to DataLoaderConfig:**
```csharp
public ErrorPolicy ErrorPolicy { get; set; } = ErrorPolicy.Continue;
public int MaxWorkerRetries { get; set; } = 3;
public TimeSpan WorkerTimeout { get; set; } = TimeSpan.FromSeconds(30);
public bool LogErrors { get; set; } = true;
```

### 4. Worker Crash Detection

**Crash Detector:**
```csharp
public class WorkerCrashDetector
{
    private readonly Dictionary<int, Task> _workerTasks;
    private readonly Dictionary<int, DateTime> _lastHeartbeat;
    private readonly TimeSpan _heartbeatTimeout;
    private readonly ErrorPolicy _errorPolicy;

    public event Action<WorkerError>? OnWorkerCrashed;
}
```

**Methods:**

**Register Worker:**
```csharp
public void RegisterWorker(int workerId, Task workerTask)
```

**Behavior:**
- Tracks worker task and start time
- Starts heartbeat monitoring

**Heartbeat Check:**
```csharp
public void UpdateHeartbeat(int workerId)
```

**Behavior:**
- Updates last heartbeat timestamp for worker
- Called periodically by workers

**Monitor Workers:**
```csharp
public async Task MonitorAsync(CancellationToken cancellationToken)
```

**Behavior:**
- Periodically checks worker task status
- Detects crashed workers (faulted or cancelled unexpectedly)
- Detects stalled workers (no heartbeat within timeout)
- Raises OnWorkerCrashed event when crash detected

### 5. Worker Recovery

**Worker Recovery Service:**
```csharp
public class WorkerRecoveryService
{
    public event Action<int, int>? OnWorkerRestarted;  // workerId, retryCount
    public event Action<int>? OnWorkerFailed;            // workerId

    public Task<bool> TryRestartWorkerAsync(int workerId, WorkerError error)
    public void MarkWorkerFailed(int workerId)
}
```

**Behavior:**

**TryRestartWorker:**
- Determines if restart is allowed based on ErrorPolicy
- Checks if retry limit exceeded
- Creates new worker task
- Returns `true` if restart successful, `false` otherwise

**MarkWorkerFailed:**
- Permanently marks worker as failed
- Prevents future restart attempts
- Raises OnWorkerFailed event

### 6. Timeout Management

**Worker Timeout Tracker:**
```csharp
public class WorkerTimeoutTracker
{
    private readonly Dictionary<int, DateTime> _operationStartTimes;
    private readonly TimeSpan _timeout;

    public event Action<int, TimeSpan>? OnWorkerTimeout;

    public void StartOperation(int workerId)
    public void EndOperation(int workerId)
    public async Task MonitorTimeoutsAsync(CancellationToken cancellationToken)
}
```

**Behavior:**

**MonitorTimeouts:**
- Continuously checks for operations exceeding timeout
- Raises OnWorkerTimeout event when timeout detected
- Can trigger worker restart or cancellation based on policy

### 7. Error Aggregator

**Error Aggregator:**
```csharp
public class ErrorAggregator
{
    private readonly List<WorkerError> _errors;
    private readonly object _lock = new object();

    public void AddError(WorkerError error)
    public IReadOnlyList<WorkerError> GetErrors()
    public WorkerError? GetLastError()
    public int GetErrorCount(int workerId)
    public void ClearErrors()
}
```

**Behavior:**
- Thread-safe collection of all errors
- Provides query methods for error statistics
- Supports filtering by worker ID

### 8. Logging Integration

**Error Logger (Optional - depends on logging framework):**
```csharp
public interface IErrorLogger
{
    void LogError(WorkerError error);
    void LogWarning(string message);
    void LogInfo(string message);
}

public class ConsoleErrorLogger : IErrorLogger
{
    // Simple implementation for now
}
```

### 9. Worker Pool Error Handling Integration

**Update WorkerPool<T> with Error Handling:**

**Add Events:**
```csharp
public event Action<WorkerError>? OnWorkerError;
public event Action<int>? OnWorkerRestarted;
public event Action<int>? OnWorkerFailed;
```

**Add Properties:**
```csharp
public ErrorAggregator ErrorAggregator { get; }
public int FailedWorkers { get; }
```

**Update Worker Task Loop:**
```csharp
while (!cancellationToken.IsCancellationRequested)
{
    try
    {
        // Update heartbeat
        crashDetector.UpdateHeartbeat(workerId);

        // Perform work
        T result = workerFunc(workerId, cancellationToken);

        // Enqueue result
        outputQueue.Enqueue(result);

        // Clear error count on success
        errorAggregator.ClearErrors();
    }
    catch (OperationCanceledException)
    {
        // Expected shutdown
        break;
    }
    catch (Exception ex)
    {
        // Handle error based on policy
        var error = new WorkerError(workerId, ex, "WorkerLoop");
        errorAggregator.AddError(error);
        OnWorkerError?.Invoke(error);

        if (errorPolicy == ErrorPolicy.Restart && retryCount < maxRetries)
        {
            retryCount++;
            // Delay before retry
            await Task.Delay(100 * retryCount, cancellationToken);
            continue;
        }
        else if (errorPolicy == ErrorPolicy.Continue)
        {
            // Skip this worker
            break;
        }
        else // FailFast
        {
            throw;
        }
    }
}
```

### 10. DataLoader Error Handling

**Add to DataLoader<T>:**

**Add Events:**
```csharp
public event Action<WorkerError>? OnWorkerError;
public event Action? OnRecoveryComplete;
public event Action? OnCriticalFailure;
```

**Add Methods:**
```csharp
public IReadOnlyList<WorkerError> GetErrors()
public ErrorAggregator GetErrorAggregator()
```

**Error Propagation:**
- Propagate worker errors to dataloader events
- On critical failure (all workers dead), raise OnCriticalFailure
- On successful recovery, raise OnRecoveryComplete

## File Structure
```
src/
  Data/
    WorkerError.cs              (Error information class)
    ErrorPolicy.cs              (Enum definition)
    WorkerCrashDetector.cs      (Crash detection)
    WorkerRecoveryService.cs    (Recovery logic)
    WorkerTimeoutTracker.cs     (Timeout management)
    ErrorAggregator.cs          (Error collection)
    IErrorLogger.cs             (Logging interface - optional)
    ConsoleErrorLogger.cs       (Simple logger - optional)
```

## Success Criteria
- [ ] Worker crashes are detected within timeout period
- [ ] Workers can be restarted successfully
- [ ] Retry limit is respected
- [ ] Different error policies work correctly
- [ ] Timeouts trigger appropriate action
- [ ] Errors are aggregated and queryable
- [ ] Events fire at appropriate times
- [ ] Recovery doesn't cause data corruption
- [ ] Critical failure stops dataloader appropriately
- [ ] Unit tests cover all error scenarios
- [ ] Unit tests verify recovery logic

## Notes
- This spec integrates with WorkerPool and DataLoader specs
- Error handling should be non-blocking where possible
- Consider exponential backoff for worker restarts
- Heartbeat interval should be configurable
- Logging is optional; provide stub implementation
- This spec enhances robustness of the entire data loading pipeline
