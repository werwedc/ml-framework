using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace MLFramework.MobileRuntime.Memory
{
    /// <summary>
    /// Default memory pool implementation with block caching and size bucketing.
    /// Uses power-of-2 buckets for efficient block reuse.
    /// </summary>
    public sealed class DefaultMemoryPool : IMemoryPool, IDisposable
    {
        private readonly object _lock = new object();
        private readonly Dictionary<long, Stack<IntPtr>> _freeBlocks;
        private readonly Dictionary<IntPtr, long> _allocatedBlocks;
        private long _totalAllocated;
        private long _memoryLimit;
        private bool _lowMemoryMode;
        private long _peakUsage;
        private int _allocationCount;
        private int _freeCount;
        private int _cacheHits;
        private int _cacheMisses;
        private bool _disposed;

        // Power-of-2 bucket sizes for efficient block reuse
        private static readonly long[] BucketSizes = GenerateBucketSizes();

        private static long[] GenerateBucketSizes()
        {
            var buckets = new List<long>();
            for (int i = 5; i <= 24; i++) // 32 bytes to 16MB
            {
                buckets.Add(1L << i);
            }
            return buckets.ToArray();
        }

        /// <summary>
        /// Creates a new memory pool with the specified initial capacity.
        /// </summary>
        /// <param name="initialCapacity">Initial capacity in bytes (default: 16MB).</param>
        public DefaultMemoryPool(long initialCapacity = 16 * 1024 * 1024)
        {
            _freeBlocks = new Dictionary<long, Stack<IntPtr>>();
            _allocatedBlocks = new Dictionary<IntPtr, long>();
            _memoryLimit = initialCapacity;
            _totalAllocated = 0;
            _peakUsage = 0;

            // Initialize empty stacks for all bucket sizes
            foreach (var size in BucketSizes)
            {
                _freeBlocks[size] = new Stack<IntPtr>();
            }
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
                    throw new ObjectDisposedException(nameof(DefaultMemoryPool));
                }

                // Round up to nearest bucket size for efficient reuse
                long bucketSize = FindBucketSize(size);
                IntPtr ptr;

                // Try to get a block from the cache
                if (_freeBlocks.ContainsKey(bucketSize) && _freeBlocks[bucketSize].Count > 0)
                {
                    ptr = _freeBlocks[bucketSize].Pop();
                    _cacheHits++;
                    _totalAllocated += bucketSize;
                }
                else
                {
                    // Allocate new block
                    ptr = AllocateNewBlock(bucketSize);
                    _cacheMisses++;
                    _totalAllocated += bucketSize;
                }

                _allocatedBlocks[ptr] = bucketSize;
                _allocationCount++;

                // Track peak usage
                if (_totalAllocated > _peakUsage)
                {
                    _peakUsage = _totalAllocated;
                }

                // Check memory limit
                if (_memoryLimit > 0 && _totalAllocated > _memoryLimit)
                {
                    throw new OutOfMemoryException(
                        $"Memory pool limit of {_memoryLimit / 1024.0 / 1024.0:F2}MB exceeded. " +
                        $"Current usage: {_totalAllocated / 1024.0 / 1024.0:F2}MB");
                }

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
                    throw new ObjectDisposedException(nameof(DefaultMemoryPool));
                }

                if (!_allocatedBlocks.ContainsKey(ptr))
                {
                    throw new InvalidOperationException(
                        $"Pointer {ptr} was not allocated from this pool or already freed");
                }

                long actualSize = _allocatedBlocks[ptr];
                _allocatedBlocks.Remove(ptr);
                _freeCount++;

                if (_lowMemoryMode)
                {
                    // In low memory mode, free immediately
                    Marshal.FreeHGlobal(ptr);
                    _totalAllocated -= actualSize;
                }
                else
                {
                    // Return to pool for reuse
                    ReturnToPool(ptr, actualSize);
                }
            }
        }

        /// <inheritdoc/>
        public void SetMemoryLimit(long maxBytes)
        {
            if (maxBytes < 0)
            {
                throw new ArgumentException("Memory limit must be non-negative", nameof(maxBytes));
            }

            lock (_lock)
            {
                _memoryLimit = maxBytes;
            }
        }

        /// <inheritdoc/>
        public long GetAvailableMemory()
        {
            lock (_lock)
            {
                // Calculate total cached memory
                long cached = 0;
                foreach (var kvp in _freeBlocks)
                {
                    cached += kvp.Value.Count * kvp.Key;
                }
                // Available is memory limit minus allocated minus cached
                return _memoryLimit - _totalAllocated + cached;
            }
        }

        /// <inheritdoc/>
        public long GetUsedMemory()
        {
            lock (_lock)
            {
                // Used is total allocated minus cached blocks
                long cached = 0;
                foreach (var kvp in _freeBlocks)
                {
                    cached += kvp.Value.Count * kvp.Key;
                }
                return _totalAllocated - cached;
            }
        }

        /// <inheritdoc/>
        public MemoryPoolStats GetStats()
        {
            lock (_lock)
            {
                long cached = 0;
                foreach (var kvp in _freeBlocks)
                {
                    cached += kvp.Value.Count * kvp.Key;
                }

                return new MemoryPoolStats
                {
                    TotalMemory = _memoryLimit > 0 ? _memoryLimit : _totalAllocated,
                    UsedMemory = _totalAllocated - cached,
                    AvailableMemory = _memoryLimit > 0 ? _memoryLimit - _totalAllocated + cached : 0,
                    AllocationCount = _allocationCount,
                    FreeCount = _freeCount,
                    CacheHits = _cacheHits,
                    CacheMisses = _cacheMisses,
                    PeakUsage = _peakUsage
                };
            }
        }

        /// <inheritdoc/>
        public void EnableLowMemoryMode(bool enable)
        {
            lock (_lock)
            {
                _lowMemoryMode = enable;

                if (enable)
                {
                    // Free all cached blocks when entering low memory mode
                    FreeAllCachedBlocks();
                }
            }
        }

        /// <inheritdoc/>
        public void PreAllocateForTensor(long size)
        {
            long bucketSize = FindBucketSize(size);

            lock (_lock)
            {
                if (_disposed)
                {
                    throw new ObjectDisposedException(nameof(DefaultMemoryPool));
                }

                // Allocate a block and immediately return it to the pool
                var ptr = AllocateNewBlock(bucketSize);
                _freeBlocks[bucketSize].Push(ptr);
                _totalAllocated += bucketSize;
            }
        }

        /// <inheritdoc/>
        public void Reset()
        {
            lock (_lock)
            {
                if (_disposed)
                {
                    throw new ObjectDisposedException(nameof(DefaultMemoryPool));
                }

                // Free all blocks
                FreeAllCachedBlocks();

                // Free all allocated blocks
                foreach (var kvp in _allocatedBlocks)
                {
                    Marshal.FreeHGlobal(kvp.Key);
                }
                _allocatedBlocks.Clear();

                // Reset statistics
                _totalAllocated = 0;
                _peakUsage = 0;
                _allocationCount = 0;
                _freeCount = 0;
                _cacheHits = 0;
                _cacheMisses = 0;
            }
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            lock (_lock)
            {
                if (!_disposed)
                {
                    Reset();
                    _disposed = true;
                }
            }
        }

        private IntPtr AllocateNewBlock(long size)
        {
            try
            {
                return Marshal.AllocHGlobal((IntPtr)size);
            }
            catch (OutOfMemoryException)
            {
                throw new OutOfMemoryException(
                    $"Failed to allocate {size / 1024.0 / 1024.0:F2}MB. " +
                    $"Pool usage: {_totalAllocated / 1024.0 / 1024.0:F2}MB / " +
                    $"{_memoryLimit / 1024.0 / 1024.0:F2}MB");
            }
        }

        private void ReturnToPool(IntPtr ptr, long size)
        {
            if (_freeBlocks.ContainsKey(size))
            {
                _freeBlocks[size].Push(ptr);
            }
            else
            {
                // If this size doesn't have a bucket, free it
                Marshal.FreeHGlobal(ptr);
                _totalAllocated -= size;
            }
        }

        private void FreeAllCachedBlocks()
        {
            foreach (var kvp in _freeBlocks)
            {
                while (kvp.Value.Count > 0)
                {
                    var ptr = kvp.Value.Pop();
                    Marshal.FreeHGlobal(ptr);
                    _totalAllocated -= kvp.Key;
                }
            }
        }

        private long FindBucketSize(long size)
        {
            // Find the smallest bucket size that can accommodate the request
            foreach (var bucketSize in BucketSizes)
            {
                if (bucketSize >= size)
                {
                    return bucketSize;
                }
            }

            // If size is larger than all buckets, return the size itself
            // This will allocate a one-off block that won't be cached
            return size;
        }
    }
}
