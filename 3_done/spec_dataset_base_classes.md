# Spec: Base Dataset Abstractions

## Overview
Define the core dataset abstractions that serve as the foundation for all data loading operations.

## Requirements

### Interfaces

#### IDataset
```csharp
public interface IDataset<T>
{
    T GetItem(int index);
    int Length { get; }
}
```

#### IIterableDataset
```csharp
public interface IIterableDataset<T>
{
    IEnumerator<T> GetEnumerator();
}
```

### Base Classes

#### MapStyleDataset
- Abstract base class implementing `IDataset<T>`
- Provides random access to data samples via index
- Must implement `GetItem(int index)` and `Length`
- Supports efficient shuffling and random sampling
- Thread-safe for read-only operations

**Key Methods:**
```csharp
public abstract class MapStyleDataset<T> : IDataset<T>
{
    public abstract T GetItem(int index);
    public abstract int Length { get; }

    protected virtual void OnDatasetCreated() { }
    protected virtual void ValidateIndex(int index) { }
}
```

#### IterableDataset
- Abstract base class implementing `IIterableDataset<T>`
- Provides sequential access to data samples
- Must implement `IEnumerator<T> GetEnumerator()`
- Suitable for streaming or infinite datasets
- Not directly compatible with random sampling

**Key Methods:**
```csharp
public abstract class IterableDataset<T> : IIterableDataset<T>
{
    public abstract IEnumerator<T> GetEnumerator();
    protected virtual void OnDatasetCreated() { }
}
```

### Error Handling
- `IndexOutOfRangeException` for invalid indices in MapStyleDataset
- `InvalidOperationException` if iterator is used incorrectly
- Validation in constructors and critical methods

### Thread Safety
- MapStyleDataset: Thread-safe for concurrent reads
- IterableDataset: Not thread-safe (single consumer expected)
- Document thread safety requirements clearly

## Acceptance Criteria
1. `IDataset<T>` interface allows random access to samples
2. `IIterableDataset<T>` interface allows sequential iteration
3. `MapStyleDataset<T>` provides index validation
4. `IterableDataset<T>` implements standard enumerator pattern
5. Both base classes call `OnDatasetCreated` during initialization
6. Unit tests cover valid/invalid indices and iteration scenarios

## Files to Create
- `src/Data/IDataset.cs`
- `src/Data/IIterableDataset.cs`
- `src/Data/MapStyleDataset.cs`
- `src/Data/IterableDataset.cs`

## Tests
- `tests/Data/MapStyleDatasetTests.cs`
- `tests/Data/IterableDatasetTests.cs`

## Notes
- Keep these classes minimal - they're foundational abstractions
- Focus on clean, extensible design
- No I/O operations in base classes - defer to implementations
