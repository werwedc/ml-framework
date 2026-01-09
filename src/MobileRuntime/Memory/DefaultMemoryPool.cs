using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using MobileRuntime.Interfaces;

namespace MobileRuntime.Memory
{
    /// <summary>
    /// Default memory pool implementation with caching
    /// </summary>
    public sealed class DefaultMemoryPool : IMemoryPool
    {
        private readonly Dictionary<long, Queue<IntPtr>> _freeBlocks;
        private readonly Dictionary<IntPtr, long> _allocatedBlocks;
        private readonly long _defaultMemoryLimit;
        private long _memoryLimit;
        private long _totalAllocated;
        private long _peakUsage;
        private int _cacheHits;
        private int _cacheMisses;
        private bool _disposed;

        public DefaultMemoryPool(long memoryLimit = 16 * 1024 * 1024)
        {
            _defaultMemoryLimit = memoryLimit;
            _memoryLimit = memoryLimit;
            _freeBlocks = new Dictionary<long, Queue<IntPtr>>();
            _allocatedBlocks = new Dictionary<IntPtr, long>();
        }

        public IntPtr Allocate(long size, DataType dataType)
        {
            long alignedSize = AlignSize(size, dataType);
            IntPtr ptr;

            // Try to reuse from cache
            if (_freeBlocks.TryGetValue(alignedSize, out var queue) && queue.Count > 0)
            {
                ptr = queue.Dequeue();
                _cacheHits++;
            }
            else
            {
                // Allocate new block
                if (_totalAllocated + alignedSize > _memoryLimit)
                {
                    throw new OutOfMemoryException(
                        $"Memory pool limit exceeded. Trying to allocate {alignedSize} bytes, " +
                        $"but {_totalAllocated} bytes are already allocated (limit: {_memoryLimit} bytes)");
                }

                ptr = Marshal.AllocHGlobal((int)alignedSize);
                _totalAllocated += alignedSize;
                _cacheMisses++;
            }

            _allocatedBlocks[ptr] = alignedSize;

            if (_totalAllocated > _peakUsage)
            {
                _peakUsage = _totalAllocated;
            }

            return ptr;
        }

        public void Free(IntPtr ptr, long size)
        {
            if (ptr == IntPtr.Zero)
                return;

            if (!_allocatedBlocks.TryGetValue(ptr, out var actualSize))
            {
                // Block wasn't allocated through this pool
                return;
            }

            _allocatedBlocks.Remove(ptr);

            // Return to cache
            if (!_freeBlocks.ContainsKey(actualSize))
            {
                _freeBlocks[actualSize] = new Queue<IntPtr>();
            }
            _freeBlocks[actualSize].Enqueue(ptr);
        }

        public MemoryPoolStats GetStats()
        {
            return new MemoryPoolStats
            {
                TotalAllocatedBytes = _totalAllocated,
                PeakUsage = _peakUsage,
                CacheHits = _cacheHits,
                CacheMisses = _cacheMisses,
                MemoryLimit = _memoryLimit,
                ActiveAllocations = _allocatedBlocks.Count
            };
        }

        public void SetMemoryLimit(long limitInBytes)
        {
            _memoryLimit = limitInBytes;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Free all allocated blocks
                    foreach (var ptr in _allocatedBlocks.Keys)
                    {
                        Marshal.FreeHGlobal(ptr);
                    }
                    _allocatedBlocks.Clear();

                    // Free all cached blocks
                    foreach (var queue in _freeBlocks.Values)
                    {
                        while (queue.Count > 0)
                        {
                            Marshal.FreeHGlobal(queue.Dequeue());
                        }
                    }
                    _freeBlocks.Clear();
                }
                _disposed = true;
            }
        }

        ~DefaultMemoryPool()
        {
            Dispose(false);
        }

        private static long AlignSize(long size, DataType dataType)
        {
            // Align to 16 bytes for SIMD compatibility
            long elementSize = GetDataTypeSize(dataType);
            long aligned = (size + 15) & ~15L;
            return Math.Max(aligned, elementSize);
        }

        private static long GetDataTypeSize(DataType dataType)
        {
            return dataType switch
            {
                DataType.Float32 => 4,
                DataType.Float16 => 2,
                DataType.Int8 => 1,
                DataType.Int16 => 2,
                DataType.Int32 => 4,
                _ => throw new ArgumentException($"Unknown data type: {dataType}")
            };
        }
    }
}
