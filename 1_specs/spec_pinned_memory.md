# Spec: Pinned Memory Utilities

## Overview
Implement pinned memory management for efficient CPU-to-GPU data transfers. Pinned memory prevents the garbage collector from moving data, enabling direct GPU access without intermediate copies.

## Requirements

### 1. IPinnedMemory<T> Interface
Defines the contract for pinned memory buffers.

```csharp
public interface IPinnedMemory<T> : IDisposable
    where T : unmanaged
{
    Span<T> Span { get; }
    IntPtr Pointer { get; }
    int Length { get; }
    bool IsPinned { get; }
    void Unpin();
}
```

### 2. PinnedMemory<T> Class
Basic pinned memory wrapper using GCHandle.

**Constructor:**
```csharp
public PinnedMemory(T[] array)
```

**Parameters:**
- `array`: The array to pin (must not be null)

**Properties:**
```csharp
public Span<T> Span { get; }     // Safe span access to array
public IntPtr Pointer { get; }   // Pointer to pinned memory
public int Length { get; }       // Array length
public bool IsPinned { get; }    // True if currently pinned
```

**Methods:**

**Unpin:**
```csharp
public void Unpin()
```

**Behavior:**
- Releases the GCHandle
- Allows GC to move array
- Can be called manually or via Dispose
- Multiple calls are safe (no-op after first)

**Dispose:**
```csharp
public void Dispose()
```

**Behavior:**
- Calls Unpin if still pinned
- Implements IDisposable pattern
- Safe to call multiple times

### 3. PinnedBuffer<T> Class
Pinned memory with automatic cleanup and pooling integration.

**Static Factory Method:**
```csharp
public static PinnedBuffer<T> Allocate(int length)
```

**Behavior:**
- Allocates new array of specified length
- Pins the array immediately
- Returns pinned buffer

**Properties:**
```csharp
public T[] Array { get; }      // Underlying array (read-only)
public Span<T> Span { get; }   // Safe span access
public IntPtr Pointer { get; } // Pointer to pinned memory
public int Length { get; }     // Buffer length
```

**Methods:**

**Copy From:**
```csharp
public void CopyFrom(T[] source, int sourceOffset = 0)
public void CopyFrom(Span<T> source)
```

**Behavior:**
- Copies data from source to pinned buffer
- Validates lengths match
- Throws `ArgumentOutOfRangeException` if invalid

**Copy To:**
```csharp
public void CopyTo(T[] destination, int destinationOffset = 0)
public void CopyTo(Span<T> destination)
```

**Behavior:**
- Copies data from pinned buffer to destination
- Validates lengths match
- Throws `ArgumentOutOfRangeException` if invalid

**Fill:**
```csharp
public void Fill(T value)
```

**Behavior:**
- Sets all elements to specified value
- Useful for zeroing buffers

### 4. PinnedMemoryPool<T>
Specialized pool for pinned memory buffers.

**Constructor:**
```csharp
public PinnedMemoryPool<T>(int bufferSize, int initialSize = 0, int maxSize = 20)
```

**Parameters:**
- `bufferSize`: Size of each buffer in the pool
- `initialSize`: Number of buffers to pre-allocate
- `maxSize`: Maximum number of buffers to keep

**Methods:**

**Rent Pinned Buffer:**
```csharp
public PinnedBuffer<T> Rent()
```

**Behavior:**
- Returns available pinned buffer from pool
- Creates new pinned buffer if pool is empty
- All buffers are already pinned and ready for GPU transfer

**Return Pinned Buffer:**
```csharp
public void Return(PinnedBuffer<T> buffer)
```

**Behavior:**
- Validates buffer size matches pool configuration
- Clears buffer contents (optional, configurable)
- Returns buffer to pool for reuse

**Resize Pool:**
```csharp
public void Resize(int newBufferSize)
```

**Behavior:**
- Clears existing pool
- Allocates buffers of new size
- Useful for changing batch sizes

### 5. PinnedMemoryHelper<T>
Static utility methods for pinned memory operations.

**Pin Array:**
```csharp
public static IPinnedMemory<T> Pin(T[] array)
```

**Behavior:**
- Creates PinnedMemory wrapper around array
- Throws `ArgumentNullException` if array is null

**Pin and Copy:**
```csharp
public static PinnedBuffer<T> PinAndCopy(T[] source)
```

**Behavior:**
- Allocates pinned buffer
- Copies source data to buffer
- Returns pinned buffer ready for GPU transfer

**Unmanaged Allocator (Advanced):**
```csharp
public static PinnedBuffer<T> AllocateUnmanaged(int length)
```

**Behavior:**
- Allocates memory using `Marshal.AllocHGlobal`
- Memory is not managed by GC
- Must be freed via Dispose
- Useful for very large buffers

### 6. Memory Pinning Strategies

**Strategy Enum:**
```csharp
public enum PinningStrategy
{
    None,              // Don't pin (fallback to regular copy)
    GCHandle,          // Use GCHandle (default, works for managed arrays)
    Unmanaged,         // Use unmanaged allocation (best for large buffers)
    PinnedObjectPool   // Use pooled pinned memory (best for reuse)
}
```

**Strategy Selector:**
```csharp
public static class PinningStrategySelector
{
    public static PinningStrategy SelectStrategy(int bufferSize, int expectedLifetime)
}
```

**Behavior:**
- Returns `None` for very small buffers (< 1KB)
- Returns `GCHandle` for medium buffers (< 1MB)
- Returns `Unmanaged` for large buffers (>= 1MB)
- Returns `PinnedObjectPool` if high reuse expected

## File Structure
```
src/
  Data/
    IPinnedMemory.cs         (Interface)
    PinnedMemory.cs          (Basic pinned wrapper)
    PinnedBuffer.cs          (Pinned buffer with utilities)
    PinnedMemoryPool.cs      (Pool for pinned buffers)
    PinnedMemoryHelper.cs    (Static utilities)
    PinningStrategy.cs       (Strategy enum and selector)
```

## Success Criteria
- [ ] Arrays can be pinned and unpinned correctly
- [ ] Pointer provides valid address to pinned memory
- [ ] Span provides safe access to array data
- [ ] Unpin allows GC to move array again
- [ ] Dispose properly cleans up resources
- [ ] CopyFrom/CopyTo correctly transfer data
- [ ] PinnedMemoryPool efficiently reuses buffers
- [ ] Helper methods are convenient and safe
- [ ] Strategy selector chooses optimal pinning method
- [ ] Unit tests verify memory is actually pinned
- [ ] Unit tests check for memory leaks

## Notes
- Use `GCHandleType.Pinned` for GCHandle-based pinning
- Use `Marshal.AllocHGlobal` and `Marshal.FreeHGlobal` for unmanaged memory
- Implement IDisposable pattern with finalizer for safety
- PinnedMemoryPool should use ArrayPool under the hood
- Consider integration with MemoryMarshal for performance
- Test with actual GPU transfer if possible (requires HAL integration)
- This spec is independent but will be used by dataloader core
