# Spec: HAL Memory Allocator Interface

## Overview
Define the memory allocator abstraction for unified memory management.

## Responsibilities
- Create IMemoryAllocator interface for different allocation strategies
- Establish contracts for allocation, deallocation, and caching

## Files to Create/Modify
- `src/HAL/IMemoryAllocator.cs` - Allocator interface
- `tests/HAL/MemoryAllocatorTests.cs` - Interface contract tests

## API Design

### IMemoryAllocator.cs
```csharp
namespace MLFramework.HAL;

/// <summary>
/// Interface for memory allocation strategies
/// </summary>
public interface IMemoryAllocator : IDisposable
{
    /// <summary>
    /// Allocate a memory buffer of the specified size
    /// </summary>
    /// <param name="size">Size in bytes</param>
    /// <returns>Memory buffer allocated on the device</returns>
    IMemoryBuffer Allocate(long size);

    /// <summary>
    /// Free a memory buffer back to the allocator
    /// </summary>
    /// <remarks>
    /// For caching allocators, this may return the buffer to a pool
    /// instead of freeing it to the OS
    /// </remarks>
    void Free(IMemoryBuffer buffer);

    /// <summary>
    /// Total size of cached memory (bytes)
    /// </summary>
    long CacheSize { get; }

    /// <summary>
    /// Total size of currently allocated memory (bytes)
    /// </summary>
    long AllocatedSize { get; }

    /// <summary>
    /// Empty the cache, freeing all unused memory back to the OS
    /// </summary>
    void EmptyCache();

    /// <summary>
    /// Device this allocator is associated with
    /// </summary>
    IDevice Device { get; }
}
```

## Testing Requirements
```csharp
public abstract class MemoryAllocatorTests
{
    protected abstract IMemoryAllocator CreateAllocator();

    [Test]
    public void Allocate_ReturnsValidBuffer()
    {
        using var allocator = CreateAllocator();
        var buffer = allocator.Allocate(1024);

        Assert.NotNull(buffer);
        Assert.AreEqual(1024, buffer.Size);
        Assert.IsTrue(buffer.IsValid);
        buffer.Dispose();
    }

    [Test]
    public void Allocate_ZeroSize_ThrowsException()
    {
        using var allocator = CreateAllocator();

        Assert.Throws<ArgumentException>(() =>
        {
            allocator.Allocate(0);
        });
    }

    [Test]
    public void Free_AllowsReuse()
    {
        using var allocator = CreateAllocator();
        var buffer1 = allocator.Allocate(1024);

        allocator.Free(buffer1);

        var buffer2 = allocator.Allocate(1024);

        Assert.NotNull(buffer2);
    }

    [Test]
    public void EmptyCache_ReducesCacheSize()
    {
        using var allocator = CreateAllocator();
        var buffer = allocator.Allocate(1024);
        allocator.Free(buffer);

        var sizeBefore = allocator.CacheSize;
        allocator.EmptyCache();
        var sizeAfter = allocator.CacheSize;

        Assert.LessOrEqual(sizeAfter, sizeBefore);
    }

    [Test]
    public void Dispose_CleansUpResources()
    {
        var allocator = CreateAllocator();
        allocator.Dispose();

        Assert.Throws<ObjectDisposedException>(() =>
        {
            allocator.Allocate(1024);
        });
    }
}
```

## Acceptance Criteria
- [ ] IMemoryAllocator interface defines all required methods
- [ ] Interface supports both simple and caching allocators
- [ ] Proper disposal pattern with IDisposable
- [ ] XML documentation for all public members
- [ ] Abstract test class for concrete allocator implementations

## Notes for Coder
- This is an interface only - no implementation required
- The abstract test class will be used by concrete allocator implementations
- Focus on clean API design
- EmptyCache() should be a no-op for non-caching allocators
- AllocatedSize includes both in-use and cached memory
