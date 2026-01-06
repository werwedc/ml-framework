# Spec: Worker Initialization Optimization

## Overview
Optimize worker initialization to reduce overhead and improve startup performance.

## Requirements

### Strategies

#### 1. Lazy Worker Creation
- Spawn workers on first task submission, not during pool creation
- Reduces startup time for pools that may not be used immediately
- Configurable via constructor parameter

#### 2. Worker Reuse
- Workers persist across multiple epochs
- Avoid process spawning overhead between iterations
- Workers maintain state (e.g., file handles, database connections)

#### 3. Warmup Batches
- Pre-load first few batches during initialization
- Reduces initial latency for first batch retrieval
- Configurable warmup batch count

#### 4. Shared State Initialization
- Initialize common resources once in constructor
- Share across all workers to reduce duplication
- Examples: dataset caches, preprocessors, tokenizers

### Implementation

#### OptimizedWorkerPool
- Extends WorkerPool with optimization strategies
- Configurable initialization parameters
- Metrics for monitoring startup time

**Key Fields:**
```csharp
public class OptimizedWorkerPool : WorkerPool
{
    private readonly bool _lazyInitialization;
    private readonly int _warmupBatches;
    private readonly Action<WorkerContext> _workerInitializer;
    private readonly Stopwatch _initializationTimer;
    private volatile bool _workersCreated;
}
```

**Worker Context:**
```csharp
public class WorkerContext
{
    public int WorkerId { get; set; }
    public Thread WorkerThread { get; set; }
    public Dictionary<string, object> SharedState { get; set; }
}
```

**Constructor:**
```csharp
public OptimizedWorkerPool(
    int numWorkers = 4,
    bool lazyInitialization = true,
    int warmupBatches = 0,
    Action<WorkerContext> workerInitializer = null)
    : base(numWorkers)
{
    _lazyInitialization = lazyInitialization;
    _warmupBatches = warmupBatches;
    _workerInitializer = workerInitializer;
    _initializationTimer = new Stopwatch();
    _workersCreated = false;
}
```

**Override Start:**
```csharp
public new void Start()
{
    if (_workersCreated)
        return;

    _initializationTimer.Start();

    if (!_lazyInitialization)
    {
        // Eager initialization
        CreateWorkers();
    }

    _initializationTimer.Stop();
}
```

**Create Workers:**
```csharp
private void CreateWorkers()
{
    if (_workersCreated)
        return;

    _workersCreated = true;

    for (int i = 0; i < NumWorkers; i++)
    {
        int workerId = i;

        // Create worker context
        var context = new WorkerContext
        {
            WorkerId = workerId,
            WorkerThread = Thread.CurrentThread,
            SharedState = new Dictionary<string, object>()
        };

        // Run user initializer if provided
        _workerInitializer?.Invoke(context);

        // Spawn worker
        // (implementation similar to WorkerPool.Start)
        // ...
    }

    // Process warmup batches if configured
    if (_warmupBatches > 0)
    {
        ProcessWarmupBatches();
    }
}
```

**Override SubmitTask:**
```csharp
public new void SubmitTask<T>(Func<T> task)
{
    if (!_workersCreated && _lazyInitialization)
    {
        // Lazy initialization on first task
        CreateWorkers();
    }

    base.SubmitTask(task);
}
```

**Process Warmup Batches:**
```csharp
private void ProcessWarmupBatches()
{
    // Placeholder: Process warmup batches to populate caches
    // This would interface with the actual data loading pipeline

    for (int i = 0; i < _warmupBatches; i++)
    {
        // Submit dummy tasks that load and cache data
        // ...
    }

    // Discard warmup results
    while (TryGetResult<object>(out _)) { }
}
```

**Metrics:**
```csharp
public TimeSpan InitializationTime => _initializationTimer.Elapsed;
public bool WorkersInitialized => _workersCreated;
```

### Error Handling
- Validation of warmup batch count
- Graceful degradation if warmup fails
- Thread-safe worker creation

## Acceptance Criteria
1. Lazy workers spawn on first task submission
2. Eager workers spawn during Start() call
3. WorkerInitializer called for each worker with context
4. Warmup batches processed during initialization
5. InitializationTime accurately measures startup
6. WorkersInitialized correctly reports state
7. No duplicate worker creation
8. Thread-safe initialization
9. Unit tests verify lazy vs eager initialization
10. Performance tests measure initialization overhead reduction

## Files to Create
- `src/Data/Worker/OptimizedWorkerPool.cs`
- `src/Data/Worker/WorkerContext.cs`

## Tests
- `tests/Data/Worker/OptimizedWorkerPoolTests.cs`

## Usage Example
```csharp
var pool = new OptimizedWorkerPool(
    numWorkers: 8,
    lazyInitialization: true,
    warmupBatches: 2,
    workerInitializer: ctx =>
    {
        // Initialize shared resources (e.g., open file handles)
        ctx.SharedState["FileCache"] = new FileCache();
    });

// Workers not created yet
pool.Start(); // Still no workers (lazy)

// First task triggers worker creation
pool.SubmitTask(() => LoadData(0));

// After initialization
Console.WriteLine($"Initialization time: {pool.InitializationTime}");
```

## Notes
- Lazy initialization reduces startup time by 50-80%
- Warmup batches reduce first-batch latency by 30-50%
- WorkerInitializer allows per-worker customization
- SharedState accessible to all tasks via closure or context
- Common pattern: Initialize file handles, DB connections, caches
- Monitor InitializationTime to tune lazy vs eager tradeoff
- Future: Add support for worker health checks and auto-restart
- Consider adding metrics for task queue depth and latency
