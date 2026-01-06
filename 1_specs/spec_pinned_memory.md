# Spec: Pinned Memory Support

## Overview
Implement pinned memory allocation for faster GPU data transfers via DMA.

## Requirements

### Interfaces

#### IPinnedMemoryAllocator
```csharp
public interface IPinnedMemoryAllocator : IDisposable
{
    IntPtr Allocate(int size);
    void Free(IntPtr pointer);
    bool IsPinnedMemorySupported { get; }
}
```

### Implementation

#### PinnedMemoryAllocator
- Uses fixed buffers or Marshal.AllocHGlobal with GC.KeepAlive
- Platform detection for CUDA availability
- Fallback to regular allocation if pinned memory unavailable
- Memory pooling for better performance

**Key Fields:**
```csharp
public class PinnedMemoryAllocator : IPinnedMemoryAllocator
{
    private readonly bool _pinnedMemorySupported;
    private readonly Dictionary<IntPtr, int> _allocatedBlocks;
    private readonly object _lock;
    private volatile bool _isDisposed;
}
```

**Constructor:**
```csharp
public PinnedMemoryAllocator(bool forcePinned = false)
{
    _lock = new object();
    _allocatedBlocks = new Dictionary<IntPtr, int>();
    _isDisposed = false;

    // Detect CUDA support (placeholder - integrate with CUDA library later)
    _pinnedMemorySupported = forcePinned || DetectCudaSupport();
}
```

**CUDA Detection (Placeholder):**
```csharp
private bool DetectCudaSupport()
{
    // Placeholder - will integrate with CUDA library
    // For now, assume CUDA is available
    return true;
}
```

**Allocate:**
```csharp
public IntPtr Allocate(int size)
{
    if (_isDisposed)
        throw new ObjectDisposedException(nameof(PinnedMemoryAllocator));

    if (size <= 0)
        throw new ArgumentOutOfRangeException(nameof(size));

    IntPtr pointer;

    if (_pinnedMemorySupported)
    {
        // Allocate pinned memory using CUDA API (placeholder)
        pointer = AllocatePinnedMemory(size);
    }
    else
        {
            // Fallback to regular allocation
            pointer = Marshal.AllocHGlobal(size);
        }

    lock (_lock)
    {
        _allocatedBlocks[pointer] = size;
    }

    return pointer;
}
```

**Allocate Pinned Memory (Placeholder):**
```csharp
private IntPtr AllocatePinnedMemory(int size)
{
    // Placeholder for CUDA cudaMallocHost / cudaHostAlloc
    // For now, use AllocHGlobal as fallback
    return Marshal.AllocHGlobal(size);
}
```

**Free:**
```csharp
public void Free(IntPtr pointer)
{
    if (_isDisposed)
        throw new ObjectDisposedException(nameof(PinnedMemoryAllocator));

    if (pointer == IntPtr.Zero)
        throw new ArgumentException("Pointer cannot be zero");

    int size;

    lock (_lock)
    {
        if (!_allocatedBlocks.TryGetValue(pointer, out size))
            throw new ArgumentException("Pointer was not allocated by this allocator");

        _allocatedBlocks.Remove(pointer);
    }

    if (_pinnedMemorySupported)
    {
        // Free pinned memory using CUDA API (placeholder)
        FreePinnedMemory(pointer);
    }
    else
        {
            Marshal.FreeHGlobal(pointer);
        }
}
```

**Free Pinned Memory (Placeholder):**
```csharp
private void FreePinnedMemory(IntPtr pointer)
{
    // Placeholder for CUDA cudaFreeHost
    // For now, use FreeHGlobal
    Marshal.FreeHGlobal(pointer);
}
```

**Properties:**
```csharp
public bool IsPinnedMemorySupported => _pinnedMemorySupported;
```

**Dispose:**
```csharp
public void Dispose()
{
    if (_isDisposed)
        return;

    _isDisposed = true;

    // Free all allocated blocks
    lock (_lock)
    {
        foreach (var kvp in _allocatedBlocks)
        {
            if (_pinnedMemorySupported)
            {
                FreePinnedMemory(kvp.Key);
            }
            else
                {
                    Marshal.FreeHGlobal(kvp.Key);
                }
        }

        _allocatedBlocks.Clear();
    }
}
```

### Helper Methods

#### CopyToPinnedMemory
```csharp
public static void CopyToPinnedMemory(IntPtr pinnedPtr, byte[] data, int offset = 0, int? length = null)
{
    if (pinnedPtr == IntPtr.Zero)
        throw new ArgumentException("Pointer cannot be zero");

    if (data == null)
        throw new ArgumentNullException(nameof(data));

    int copyLength = length ?? data.Length;

    if (offset < 0 || offset >= data.Length)
        throw new ArgumentOutOfRangeException(nameof(offset));

    if (copyLength < 0 || offset + copyLength > data.Length)
        throw new ArgumentOutOfRangeException(nameof(length));

    Marshal.Copy(data, offset, pinnedPtr, copyLength);
}
```

## Acceptance Criteria
1. Allocator correctly detects CUDA support
2. Allocate returns valid pointer for pinned memory
3. Free correctly deallocates memory
4. Allocator tracks all allocated blocks
5. Dispose frees all allocated memory
6. Fallback to regular allocation if pinned memory unavailable
7. CopyToPinnedMemory correctly copies data
8. Thread-safe for concurrent allocate/free operations
9. Unit tests verify memory tracking and cleanup
10. Integration tests with actual CUDA when available

## Files to Create
- `src/Data/Memory/IPinnedMemoryAllocator.cs`
- `src/Data/Memory/PinnedMemoryAllocator.cs`

## Tests
- `tests/Data/Memory/PinnedMemoryAllocatorTests.cs`

## Usage Example
```csharp
using (var allocator = new PinnedMemoryAllocator())
{
    if (allocator.IsPinnedMemorySupported)
    {
        int size = 1024 * 1024; // 1 MB
        IntPtr pinnedPtr = allocator.Allocate(size);

        // Copy data to pinned memory
        byte[] data = new byte[size];
        PinnedMemoryAllocator.CopyToPinnedMemory(pinnedPtr, data);

        // Use pinned memory for GPU transfer (placeholder)
        GpuTransfer(pinnedPtr, size);

        allocator.Free(pinnedPtr);
    }
}
```

## Notes
- Pinned memory allows asynchronous DMA transfers over PCIe
- Critical for overlapping data transfer with GPU computation
- Placeholder implementation uses AllocHGlobal for now
- Will integrate with actual CUDA library (cudaMallocHost/cudaFreeHost)
- Memory tracking prevents double-free and leaks
- Consider adding memory pooling for small allocations
- Monitor pinned memory usage to avoid system memory pressure
- On systems without CUDA, falls back to regular allocation
- Future: add support for unified memory (cudaMallocManaged)
