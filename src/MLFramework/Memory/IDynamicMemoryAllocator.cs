using System;
using System.Collections.Generic;

namespace MLFramework.Memory;

/// <summary>
/// Interface for dynamic memory allocator.
/// </summary>
public interface IDynamicMemoryAllocator
{
    /// <summary>
    /// Allocates memory for a tensor with given shape bounds.
    /// </summary>
    IMemoryHandle Allocate(ShapeBounds bounds);

    /// <summary>
    /// Resizes an existing memory allocation.
    /// </summary>
    void Resize(IMemoryHandle handle, int[] newShape);

    /// <summary>
    /// Frees an allocated memory handle.
    /// </summary>
    void Free(IMemoryHandle handle);

    /// <summary>
    /// Gets current allocation statistics.
    /// </summary>
    AllocationStats GetAllocationStats();
}
