# Spec: HAL Caching Memory Allocator

## Overview
Implement a block-based caching allocator to minimize memory allocation overhead.

## Responsibilities
- Create CachingAllocator implementing IMemoryAllocator
- Implement block splitting/merging to reduce fragmentation
- Track allocated and cached blocks
- Provide memory statistics

## Files to Create/Modify
- `src/HAL/CachingAllocator.cs` - Caching allocator implementation
- `tests/HAL/CachingAllocatorTests.cs` - Caching allocator tests

## API Design

### CachingAllocator.cs
```csharp
namespace MLFramework.HAL;

/// <summary>
/// Block-based caching allocator that pools memory blocks for reuse
/// </summary>
public class CachingAllocator : IMemoryAllocator
{
    private readonly IDevice _device;
    private readonly SortedList<long, LinkedList<MemoryBlock>> _freeBlocks;
    private readonly List<MemoryBlock> _allocatedBlocks;
    private readonly object _lock = new();

    public long CacheSize { get; private set; }
    public long AllocatedSize { get; private set; }
    public IDevice Device => _device;

    public CachingAllocator(IDevice device)
    {
        _device = device;
        _freeBlocks = new SortedList<long, LinkedList<MemoryBlock>>();
        _allocatedBlocks = new List<MemoryBlock>();
    }

    public IMemoryBuffer Allocate(long size)
    {
        lock (_lock)
        {
            // Round up to 16-byte alignment
            var alignedSize = AlignSize(size);

            // Try to find a suitable free block
            var block = FindOrCreateBlock(alignedSize);

            _allocatedBlocks.Add(block);
            AllocatedSize += alignedSize;

            return new CachedMemoryBuffer(block, this);
        }
    }

    public void Free(IMemoryBuffer buffer)
    {
        if (buffer == null)
            throw new ArgumentNullException(nameof(buffer));

        lock (_lock)
        {
            if (buffer is CachedMemoryBuffer cachedBuffer)
            {
                var block = cachedBuffer.Block;

                if (!_allocatedBlocks.Remove(block))
                    throw new ArgumentException("Buffer not allocated by this allocator");

                AllocatedSize -= block.Size;
                AddToFreeList(block);
                TryMergeBlocks();
            }
            else
            {
                throw new ArgumentException("Invalid buffer type");
            }
        }
    }

    public void EmptyCache()
    {
        lock (_lock)
        {
            foreach (var size in _freeBlocks.Keys.ToList())
            {
                foreach (var block in _freeBlocks[size].ToList())
                {
                    Marshal.FreeHGlobal(block.Pointer);
                    CacheSize -= block.Size;
                }
                _freeBlocks[size].Clear();
            }
        }
    }

    private MemoryBlock FindOrCreateBlock(long size)
    {
        // Try to find a block of exact size
        if (_freeBlocks.TryGetValue(size, out var blocks) && blocks.Count > 0)
        {
            return blocks.First.Value;
        }

        // Try to find a larger block and split it
        foreach (var entry in _freeBlocks)
        {
            if (entry.Key >= size)
            {
                var block = entry.Value.First.Value;
                entry.Value.RemoveFirst();

                if (entry.Key > size)
                {
                    // Split the block
                    var remainingBlock = new MemoryBlock(
                        block.Pointer + size,
                        entry.Key - size);
                    AddToFreeList(remainingBlock);
                    return new MemoryBlock(block.Pointer, size);
                }
                return block;
            }
        }

        // No suitable free block, allocate new
        return AllocateNewBlock(size);
    }

    private MemoryBlock AllocateNewBlock(long size)
    {
        var pointer = Marshal.AllocHGlobal((IntPtr)size);
        CacheSize += size;

        return new MemoryBlock(pointer, size);
    }

    private void AddToFreeList(MemoryBlock block)
    {
        if (!_freeBlocks.TryGetValue(block.Size, out var blocks))
        {
            blocks = new LinkedList<MemoryBlock>();
            _freeBlocks[block.Size] = blocks;
        }
        blocks.AddLast(block);
    }

    private void TryMergeBlocks()
    {
        // Simple implementation: merge adjacent blocks
        // Full implementation would maintain ordered list for O(1) merging
        foreach (var size in _freeBlocks.Keys.ToList())
        {
            var blocks = _freeBlocks[size];
            if (blocks.Count > 1)
            {
                var first = blocks.First.Value;
                var second = blocks.First.Next?.Value;

                if (second != null && first.Pointer + first.Size == second.Pointer)
                {
                    blocks.RemoveFirst();
                    blocks.RemoveFirst();

                    var merged = new MemoryBlock(first.Pointer, first.Size + second.Size);
                    AddToFreeList(merged);
                }
            }
        }
    }

    private long AlignSize(long size)
    {
        return (size + 15) & ~15;
    }

    public void Dispose()
    {
        EmptyCache();
    }

    private class MemoryBlock
    {
        public IntPtr Pointer { get; }
        public long Size { get; }

        public MemoryBlock(IntPtr pointer, long size)
        {
            Pointer = pointer;
            Size = size;
        }
    }

    private class CachedMemoryBuffer : IMemoryBuffer
    {
        private readonly CachingAllocator _allocator;
        private bool _disposed;

        public MemoryBlock Block { get; }
        public IntPtr Pointer => Block.Pointer;
        public long Size => Block.Size;
        public IDevice Device => _allocator.Device;
        public bool IsValid => !_disposed;

        public CachedMemoryBuffer(MemoryBlock block, CachingAllocator allocator)
        {
            Block = block;
            _allocator = allocator;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _allocator.Free(this);
                _disposed = true;
            }
        }
    }
}
```

## Testing Requirements
```csharp
public class CachingAllocatorTests : MemoryAllocatorTests
{
    private Mock<IDevice>? _mockDevice;

    protected override IMemoryAllocator CreateAllocator()
    {
        _mockDevice = new Mock<IDevice>();
        _mockDevice.Setup(d => d.DeviceType).Returns(DeviceType.CPU);
        return new CachingAllocator(_mockDevice.Object);
    }

    [Test]
    public void Allocate_AlignsSize()
    {
        using var allocator = CreateAllocator();
        var buffer = allocator.Allocate(100);

        Assert.AreEqual(112, buffer.Size); // 100 rounded to 16-byte alignment
    }

    [Test]
    public void Free_ReuseBuffer()
    {
        using var allocator = CreateAllocator();
        var buffer1 = allocator.Allocate(1024);
        var ptr1 = buffer1.Pointer;

        allocator.Free(buffer1);
        var buffer2 = allocator.Allocate(1024);
        var ptr2 = buffer2.Pointer;

        Assert.AreEqual(ptr1, ptr2);
    }

    [Test]
    public void CacheSize_Accumulates()
    {
        using var allocator = CreateAllocator();
        var buffer1 = allocator.Allocate(1024);
        allocator.Free(buffer1);

        var buffer2 = allocator.Allocate(512);
        allocator.Free(buffer2);

        Assert.Greater(allocator.CacheSize, 0);
    }

    [Test]
    public void EmptyCache_ResetsCacheSize()
    {
        using var allocator = CreateAllocator();
        var buffer1 = allocator.Allocate(1024);
        allocator.Free(buffer1);

        allocator.EmptyCache();

        Assert.AreEqual(0, allocator.CacheSize);
    }
}
```

## Acceptance Criteria
- [ ] CachingAllocator implements IMemoryAllocator
- [ ] Memory blocks are cached and reused
- [ ] Block splitting works (large blocks split to fit requests)
- [ ] Block merging works (adjacent free blocks combined)
- [ ] Memory alignment to 16-byte boundaries
- [ ] All tests pass
- [ ] Thread-safe with proper locking

## Notes for Coder
- This is a simplified caching allocator
- Full implementation would have more sophisticated fragmentation handling
- SortedList is used for O(log n) block lookup
- Block merging is simplified - real implementation would be more robust
- Ensure thread safety with proper locking
