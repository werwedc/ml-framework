# Spec: Storage Backend Interface

## Overview
Define a storage abstraction layer that allows the visualization system to use different storage backends (file-based, in-memory, remote) for event persistence.

## Objectives
- Create abstract storage backend interface
- Support multiple storage implementations through polymorphism
- Enable pluggable storage backends
- Define common storage operations for all backend types

## API Design

```csharp
// Storage backend interface
public interface IStorageBackend
{
    // Lifecycle
    void Initialize(string connectionString);
    void Shutdown();
    bool IsInitialized { get; }

    // Event storage
    void StoreEvent(Event eventData);
    Task StoreEventAsync(Event eventData);
    void StoreEvents(IEnumerable<Event> events);
    Task StoreEventsAsync(IEnumerable<Event> events);

    // Event retrieval (optional for some backends)
    IEnumerable<Event> GetEvents(long startStep, long endStep);
    Task<IEnumerable<Event>> GetEventsAsync(long startStep, long endStep);

    // Storage management
    void Flush();
    Task FlushAsync();
    long EventCount { get; }
    void Clear();
}

// Storage backend factory
public interface IStorageBackendFactory
{
    IStorageBackend CreateBackend(string backendType, string connectionString);
}

// Configuration for storage backends
public class StorageConfiguration
{
    public string BackendType { get; set; } // "file", "memory", "remote"
    public string ConnectionString { get; set; }
    public int WriteBufferSize { get; set; } = 100;
    public TimeSpan FlushInterval { get; set; } = TimeSpan.FromSeconds(1);
    public bool EnableAsyncWrites { get; set; } = true;
}
```

## Implementation Requirements

### 1. Core Interfaces (30-45 min)
- Define `IStorageBackend` interface with all methods
- Define `IStorageBackendFactory` interface
- Create `StorageConfiguration` class with properties
- Add validation for connection strings and configurations

### 2. Base Storage Implementation (30-45 min)
- Create abstract `StorageBackendBase` class with common functionality:
  - Write buffering logic
  - Async flush timer
  - Basic event counting
  - Initialization state management
- Implement thread-safe buffer management
- Add automatic flush on buffer size threshold
- Add automatic flush on time interval
- Ensure flush is called on disposal

### 3. Storage Factory (20-30 min)
- Implement `StorageBackendFactory` class
- Support registration of custom backend types
- Provide mapping from string to concrete backend types
- Include error handling for unknown backend types

## File Structure
```
src/
  MLFramework.Visualization/
    Storage/
      IStorageBackend.cs
      StorageBackendBase.cs
      IStorageBackendFactory.cs
      StorageBackendFactory.cs
      StorageConfiguration.cs

tests/
  MLFramework.Visualization.Tests/
    Storage/
      StorageBackendBaseTests.cs
      StorageBackendFactoryTests.cs
```

## Dependencies
- `MLFramework.Visualization.Events` (Event types)

## Integration Points
- Used by Event Collector to persist events
- Different backends will be implemented in subsequent specs
- Configured by Visualizer main API

## Success Criteria
- Interface is clean and flexible enough for multiple backend types
- Base implementation provides useful shared functionality
- Factory pattern allows easy addition of new backends
- Unit tests verify buffer management and flush behavior
