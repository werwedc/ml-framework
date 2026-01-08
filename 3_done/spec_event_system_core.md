# Spec: Event System Core

## Overview
Implement a publish-subscribe event system to collect events from the ML framework for visualization and profiling purposes. This is the foundational component that all other visualization features depend on.

## Objectives
- Create a thread-safe event system for collecting framework events
- Support multiple event types (metrics, histograms, graphs, profiling)
- Enable subscription to specific event types
- Provide async event publishing to minimize performance impact

## API Design

```csharp
// Event types
public enum EventType
{
    ScalarMetric,
    Histogram,
    ComputationalGraph,
    ProfilingStart,
    ProfilingEnd,
    MemoryAllocation,
    TensorOperation,
    Custom
}

// Base event class
public abstract class Event
{
    public EventType Type { get; }
    public DateTime Timestamp { get; }
    public long Step { get; }
    public Dictionary<string, string> Metadata { get; }
}

// Scalar metric event
public class ScalarMetricEvent : Event
{
    public string Name { get; }
    public float Value { get; }
}

// Publisher interface
public interface IEventPublisher
{
    void Publish<T>(T eventData) where T : Event;
    Task PublishAsync<T>(T eventData) where T : Event;
}

// Subscriber interface
public interface IEventSubscriber
{
    void Subscribe<T>(Action<T> handler) where T : Event;
    void Unsubscribe<T>(Action<T> handler) where T : Event;
    void SubscribeAll(Action<Event> handler);
}

// Core event system
public class EventSystem : IEventPublisher, IEventSubscriber
{
    public EventSystem(bool enableAsync = true);
    void Shutdown();
}
```

## Implementation Requirements

### 1. Event Base Classes (30-45 min)
- Create `Event` abstract base class with common properties
- Implement concrete event types:
  - `ScalarMetricEvent`
  - `HistogramEvent` (placeholder for now)
  - `ComputationalGraphEvent` (placeholder)
  - `ProfilingStartEvent` and `ProfilingEndEvent`
  - `MemoryAllocationEvent`
  - `TensorOperationEvent`
- Add metadata dictionary for extensibility

### 2. Event System Core (45-60 min)
- Implement thread-safe publish-subscribe pattern
- Use `ConcurrentDictionary` for managing subscribers
- Support both sync and async publishing:
  - Sync: Direct invocation for critical events
  - Async: Task-based for non-blocking logging
- Implement event filtering by type
- Add shutdown/cleanup logic
- Handle exceptions in subscribers without affecting other subscribers

### 3. Event Factory (20-30 min)
- Create factory methods for common event creation
- Auto-generate timestamps and step numbers
- Support event chaining for complex scenarios

## File Structure
```
src/
  MLFramework.Visualization/
    Events/
      Event.cs
      EventTypes.cs
      ScalarMetricEvent.cs
      HistogramEvent.cs
      ProfilingEvent.cs
      IEventPublisher.cs
      IEventSubscriber.cs
      EventSystem.cs
      EventFactory.cs

tests/
  MLFramework.Visualization.Tests/
    Events/
      EventSystemTests.cs
      EventFactoryTests.cs
```

## Dependencies
- None (core system component)

## Integration Points
- Used by all visualization components
- Will be integrated with tensor operations, training loops, memory allocator

## Success Criteria
- Event system can handle 10,000+ events/second with <1ms latency
- Thread-safe under concurrent access (10+ publishers, 5+ subscribers)
- Async publishing doesn't block training loop
- Unit tests verify thread-safety and correct event delivery
