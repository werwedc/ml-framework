# Spec: Async Event Collector

## Overview
Implement an asynchronous event collector that buffers events and flushes them in the background to minimize the impact on training performance.

## Objectives
- Collect events asynchronously without blocking the main thread
- Provide buffering and batch processing
- Support configurable flush strategies (time-based, size-based, manual)
- Handle backpressure when system is overloaded

## API Design

```csharp
// Flush strategy
public enum FlushStrategy
{
    TimeBased,      // Flush on timer interval
    SizeBased,      // Flush when buffer reaches size
    Manual,         // Flush only when explicitly requested
    Hybrid          // Combination of time and size
}

// Event collector configuration
public class EventCollectorConfig
{
    public int BufferCapacity { get; set; } = 1000;
    public TimeSpan FlushInterval { get; set; } = TimeSpan.FromSeconds(1);
    public int BatchSize { get; set; } = 100;
    public FlushStrategy Strategy { get; set; } = FlushStrategy.Hybrid;
    public bool EnableBackpressure { get; set; } = true;
    public int MaxQueueLength { get; set; } = 10000;
}

// Event collector
public interface IEventCollector
{
    // Configuration
    EventCollectorConfig Config { get; set; }

    // Event collection
    void Collect<T>(T eventData) where T : Event;
    Task CollectAsync<T>(T eventData) where T : Event;

    // Flush control
    void Flush();
    Task FlushAsync();

    // Lifecycle
    void Start();
    void Stop();
    bool IsRunning { get; }
    int PendingEventCount { get; }
}

public class AsyncEventCollector : IEventCollector
{
    public AsyncEventCollector(IStorageBackend storage, EventCollectorConfig config = null);
    public AsyncEventCollector(IEnumerable<IEventSubscriber> subscribers, EventCollectorConfig config = null);
}
```

## Implementation Requirements

### 1. EventBuffer (30-45 min)
- Implement thread-safe buffer for events:
  - Use `ConcurrentQueue<Event>` or `Channel<Event>`
  - Support bounded capacity (drop oldest when full)
  - Provide efficient enqueue/dequeue operations
- Implement batch dequeue:
  - Dequeue up to `BatchSize` events at once
  - Block until events available or timeout
- Track buffer statistics:
  - Current size
  - Peak size
  - Dropped events count

### 2. EventCollector Core (45-60 min)
- Implement `IEventCollector` interface
- Create background task for processing events:
  - Dequeue events from buffer
  - Forward to storage backend or subscribers
  - Handle exceptions gracefully
- Implement flush strategies:
  - Time-based: Use `Timer` or `Task.Delay`
  - Size-based: Check buffer size periodically
  - Hybrid: Combine both approaches
- Implement backpressure handling:
  - When buffer full, block or drop events based on config
  - Provide metrics for monitoring backpressure
- Support graceful shutdown:
  - Stop accepting new events
  - Flush remaining events
  - Wait for background task to complete

### 3. AsyncEventCollector (30-45 min)
- Inherit from `EventCollector` or implement directly
- Support multiple destinations:
  - Storage backend
  - Multiple event subscribers
- Route events to appropriate destination based on type
- Implement async/await throughout for non-blocking operations
- Use `CancellationToken` for cooperative cancellation

### 4. Performance Optimizations (30-45 min)
- Use value tasks where possible to reduce allocations
- Implement object pooling for common event types
- Batch events before sending to storage:
  - Group multiple events into single storage operation
  - Reduce overhead of individual writes
- Use async I/O operations for file storage
- Minimize locks and contention points
- Implement prioritization for critical events

## File Structure
```
src/
  MLFramework.Visualization/
    Collection/
      EventBuffer.cs
      IEventCollector.cs
      EventCollector.cs
      AsyncEventCollector.cs
      Configuration/
        EventCollectorConfig.cs
      Performance/
        ObjectPool.cs
        BatchProcessor.cs

tests/
  MLFramework.Visualization.Tests/
    Collection/
      EventCollectorTests.cs
      AsyncEventCollectorTests.cs
      PerformanceTests.cs
```

## Dependencies
- `MLFramework.Visualization.Events` (Event types)
- `MLFramework.Visualization.Storage` (Storage backend)

## Integration Points
- Used by TensorBoardVisualizer to collect events
- Reduces overhead of logging during training
- Can be configured to balance between throughput and latency

## Success Criteria
- Collecting 10,000 events/second with <1ms overhead per event
- Background thread consumes <5% CPU when idle
- Graceful shutdown completes in <100ms
- No events lost during normal operation
- Backpressure handling prevents memory bloat
- Unit tests verify thread-safety and correct event delivery
- Performance tests verify minimal overhead
