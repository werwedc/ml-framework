namespace MLFramework.HAL.CUDA;

/// <summary>
/// Memory allocator for CUDA devices with caching support
/// </summary>
public class CudaAllocator : IMemoryAllocator
{
    private readonly CudaDevice _device;
    private readonly CudaCachingAllocator _cachingAllocator;

    public IDevice Device => _device;
    public long CacheSize => _cachingAllocator.CacheSize;
    public long AllocatedSize => _cachingAllocator.AllocatedSize;

    public CudaAllocator(CudaDevice device)
    {
        _device = device;
        _cachingAllocator = new CudaCachingAllocator(device);
    }

    public IMemoryBuffer Allocate(long size)
    {
        // Allocate via caching allocator
        var buffer = _cachingAllocator.Allocate(size);

        // Set memory to zero (CUDA memset)
        CudaException.CheckError(
            CudaApi.CudaMemset(buffer.Pointer, 0, (ulong)size));

        return buffer;
    }

    public void Free(IMemoryBuffer buffer)
    {
        _cachingAllocator.Free(buffer);
    }

    public void EmptyCache()
    {
        _cachingAllocator.EmptyCache();
    }

    public void Dispose()
    {
        _cachingAllocator.Dispose();
    }

    /// <summary>
    /// CUDA-specific caching allocator that uses CUDA memory APIs
    /// </summary>
    private class CudaCachingAllocator
    {
        private readonly CudaDevice _device;
        private readonly SortedList<long, LinkedList<MemoryBlock>> _freeBlocks;
        private readonly List<MemoryBlock> _allocatedBlocks;
        private readonly object _lock = new();

        public long CacheSize { get; private set; }
        public long AllocatedSize { get; private set; }

        public CudaCachingAllocator(CudaDevice device)
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
                        CudaException.CheckError(
                            CudaApi.CudaFree(block.Pointer));
                        CacheSize -= block.Size;
                    }
                    _freeBlocks[size].Clear();
                }
            }
        }

        public void Dispose()
        {
            EmptyCache();
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
                            new IntPtr(block.Pointer.ToInt64() + size),
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
            CudaException.CheckError(
                CudaApi.CudaMalloc(out IntPtr pointer, (ulong)size));

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
            foreach (var size in _freeBlocks.Keys.ToList())
            {
                var blocks = _freeBlocks[size];
                if (blocks.Count > 1)
                {
                    var first = blocks.First.Value;
                    var second = blocks.First.Next?.Value;

                    if (second != null && new IntPtr(first.Pointer.ToInt64() + first.Size) == second.Pointer)
                    {
                        blocks.RemoveFirst();
                        blocks.RemoveFirst();

                        var merged = new MemoryBlock(first.Pointer, first.Size + second.Size);
                        AddToFreeList(merged);
                    }
                }
            }
        }

        private static long AlignSize(long size)
        {
            return (size + 15) & ~15;
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
            private readonly CudaCachingAllocator _allocator;
            private bool _disposed;

            public MemoryBlock Block { get; }
            public IntPtr Pointer => Block.Pointer;
            public long Size => Block.Size;
            public IDevice Device => _allocator._device;
            public bool IsValid => !_disposed;

            public CachedMemoryBuffer(MemoryBlock block, CudaCachingAllocator allocator)
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
}
