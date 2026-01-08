# Spec: Dataset Interface and Basic Implementations

## Overview
Define the fundamental dataset abstraction that provides a uniform interface for data sources. This spec establishes the contract for all datasets in the ML framework.

## Requirements

### 1. IDataset<T> Interface
Create a generic interface that represents a collection of data items.

**Public Members:**
- `int Count { get; }` - Returns total number of items in the dataset
- `T GetItem(int index)` - Retrieves a single item by index
- `T[] GetBatch(int[] indices)` - (Optional) Retrieves multiple items at once for efficiency

**Behavior:**
- Must throw `ArgumentOutOfRangeException` if index is out of bounds
- Must be thread-safe for concurrent reads (readers can call GetItem in parallel)
- Should support negative indexing (e.g., -1 returns last item)

### 2. ListDataset<T> Implementation
Simple in-memory dataset wrapper around `IList<T>` or arrays.

**Constructor:**
```csharp
public ListDataset(IList<T> items)
```

**Implementation Details:**
- Store items in a readonly field
- Validate non-null collection in constructor
- Implement `Count` by delegating to `items.Count`
- Implement `GetItem` with index bounds checking

### 3. ArrayDataset<T> Implementation
Optimized implementation specifically for arrays to avoid interface overhead.

**Constructor:**
```csharp
public ArrayDataset(T[] items)
```

**Implementation Details:**
- Store items in a readonly field
- Validate non-null array in constructor
- Implement `Count` as `items.Length`
- Direct array access in `GetItem`

### 4. InMemoryDataset<T> (Generic Convenience Class)
Provides a static factory method for easy dataset creation from collections.

**Factory Method:**
```csharp
public static InMemoryDataset<T> FromEnumerable(IEnumerable<T> items)
```

**Behavior:**
- Materializes enumerable into array
- Returns `ArrayDataset<T>` for arrays
- Returns `ListDataset<T>` for other collections

### 5. Dataset<T> Abstract Base Class
Optional base class providing common functionality.

**Protected Members:**
- `virtual void ValidateIndex(int index)` - Shared index validation logic
- `virtual T NormalizeIndex(int index)` - Handles negative indexing

## File Structure
```
src/
  Data/
    IDataset.cs           (Interface)
    Dataset.cs            (Abstract base class)
    ListDataset.cs        (List implementation)
    ArrayDataset.cs       (Array implementation)
    InMemoryDataset.cs    (Factory class)
```

## Success Criteria
- [ ] Interface defined with Count and GetItem members
- [ ] ListDataset wraps IList<T> correctly
- [ ] ArrayDataset optimizes array access
- [ ] Negative indexing works correctly (-1 returns last item)
- [ ] IndexOutOfRangeException thrown for invalid indices
- [ ] All classes are thread-safe for concurrent reads
- [ ] Unit tests cover all implementations

## Notes
- This is a foundational component; all other dataloader specs depend on this
- Keep implementations minimal; more complex datasets (ImageDataset, etc.) will be separate specs
- Focus on interface design; performance optimizations can come later
