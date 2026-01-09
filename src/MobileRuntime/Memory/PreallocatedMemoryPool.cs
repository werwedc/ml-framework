using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace MLFramework.MobileRuntime.Memory
{
    /// <summary>
    /// Pre-allocated memory pool that manages a single contiguous memory block.
    /// Uses a simple bump pointer allocator for fast allocation.
    /// </summary>
    public sealed class PreallocatedMemoryPool : IMemoryPool, IDisposable
    {
        private readonly IntPtr _baseAddress;
        private readonly long _totalSize;
        private readonly List<MemoryBlock> _blocks;
        private readonly object _lock = new object();
        private long _offset;
        private bool _disposed;

        private struct MemoryBlock
        {
            public IntPtr Address;
            public long Size;
            public bool InUse;
        }

        /// <summary>
        /// Creates a new pre-allocated memory pool.
        /// </summary>
        /// <param name="totalSize">Total size of the memory pool in bytes.</param>
        public PreallocatedMemoryPool(long totalSize)
        {
            if (totalSize <= 0)
            {
                throw new ArgumentException("Total size must be positive", nameof(totalSize));
            }

            _totalSize = totalSize;
            _baseAddress = Marshal.AllocHGlobal((IntPtr)totalSize);
            _offset = 0;
            _blocks = new List<MemoryBlock>();
            _disposed = false;
        }

        /// <inheritdoc/>
        public IntPtr Allocate(long size, DataType dataType)
        {
            if (size <= 0)
            {
                throw new ArgumentException("Size must be positive", nameof(size));
            }

            lock (_lock)
            {
                if (_disposed)
                {
                    throw new ObjectDisposedException(nameof(PreallocatedMemoryPool));
                }

                // Align to 16-byte boundary for performance
                long alignedSize = AlignTo(size, 16);

                // Check if there's enough space
                if (_offset + alignedSize > _totalSize)
                {
                    throw new OutOfMemoryException(
                        $"Pre-allocated pool exhausted. " +
                        $"Requested: {alignedSize} bytes, " +
                        $"Available: {_totalSize - _offset} bytes");
                }

                // Allocate block
                IntPtr ptr = new IntPtr(_baseAddress.ToInt64() + _offset);
                _blocks.Add(new MemoryBlock
                {
                    Address = ptr,
                    Size = alignedSize,
                    InUse = true
                });
                _offset += alignedSize;

                return ptr;
            }
        }

        /// <inheritdoc/>
        public void Free(IntPtr ptr, long size)
        {
            if (ptr == IntPtr.Zero)
            {
                return;
            }

            lock (_lock)
            {
                if (_disposed)
                {
                    throw new ObjectDisposedException(nameof(PreallocatedMemoryPool));
                }

                // Find the block and mark as free
                for (int i = 0; i < _blocks.Count; i++)
                {
                    if (_blocks[i].Address == ptr)
                    {
                        if (!_blocks[i].InUse)
                        {
                            throw new InvalidOperationException($"Block at {ptr} is already free");
                        }
                        var block = _blocks[i];
                        block.InUse = false;
                        _blocks[i] = block;
                        return;
                    }
                }

                throw new InvalidOperationException($"Pointer {ptr} was not allocated from this pool");
            }
        }

        /// <inheritdoc/>
        public void SetMemoryLimit(long maxBytes)
        {
            lock (_lock)
            {
                if (maxBytes > _totalSize)
                {
                    throw new InvalidOperationException(
                        $"Cannot set memory limit larger than pre-allocated size ({_totalSize} bytes)");
                }

                // This pool has a fixed size, so we just track the limit
                // but don't actually enforce it differently
            }
        }

        /// <inheritdoc/>
        public long GetAvailableMemory()
        {
            lock (_lock)
            {
                return _totalSize - _offset;
            }
        }

        /// <inheritdoc/>
        public long GetUsedMemory()
        {
            lock (_lock)
            {
                long used = 0;
                foreach (var block in _blocks)
                {
                    if (block.InUse)
                    {
                        used += block.Size;
                    }
                }
                return used;
            }
        }

        /// <inheritdoc/>
        public MemoryPoolStats GetStats()
        {
            lock (_lock)
            {
                long used = 0;
                int allocatedBlocks = 0;
                foreach (var block in _blocks)
                {
                    if (block.InUse)
                    {
                        used += block.Size;
                        allocatedBlocks++;
                    }
                }

                return new MemoryPoolStats
                {
                    TotalMemory = _totalSize,
                    UsedMemory = used,
                    AvailableMemory = _totalSize - _offset,
                    AllocationCount = _blocks.Count,
                    FreeCount = _blocks.Count - allocatedBlocks,
                    CacheHits = 0, // Pre-allocated pool doesn't use caching
                    CacheMisses = _blocks.Count,
                    PeakUsage = _offset // Peak usage is the highest offset reached
                };
            }
        }

        /// <inheritdoc/>
        public void EnableLowMemoryMode(bool enable)
        {
            // Pre-allocated pool doesn't have a low memory mode
            // It always uses the pre-allocated memory efficiently
        }

        /// <inheritdoc/>
        public void PreAllocateForTensor(long size)
        {
            // Pre-allocation is automatic in this pool
            // No action needed
        }

        /// <inheritdoc/>
        public void Reset()
        {
            lock (_lock)
            {
                if (_disposed)
                {
                    throw new ObjectDisposedException(nameof(PreallocatedMemoryPool));
                }

                _blocks.Clear();
                _offset = 0;
            }
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            lock (_lock)
            {
                if (!_disposed)
                {
                    if (_baseAddress != IntPtr.Zero)
                    {
                        Marshal.FreeHGlobal(_baseAddress);
                    }
                    _disposed = true;
                }
            }
        }

        private long AlignTo(long size, long alignment)
        {
            return (size + alignment - 1) & ~(alignment - 1);
        }
    }
}
