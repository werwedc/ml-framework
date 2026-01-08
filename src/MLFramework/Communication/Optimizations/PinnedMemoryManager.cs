using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using MLFramework.Communication;
using MLFramework.Distributed.Communication;
using RitterFramework.Core.Tensor;

namespace MLFramework.Communication.Optimizations
{
    /// <summary>
    /// Manages pinned memory allocations for efficient transfers
    /// </summary>
    public class PinnedMemoryManager : IDisposable
    {
        private readonly Dictionary<IntPtr, int> _allocations;
        private readonly object _lock;
        private readonly int _maxPinnedBytes;
        private int _totalPinnedBytes;
        private bool _disposed;

        /// <summary>
        /// Maximum number of bytes that can be pinned
        /// </summary>
        public int TotalPinnedBytes => _totalPinnedBytes;

        /// <summary>
        /// Create a pinned memory manager
        /// </summary>
        /// <param name="maxPinnedBytes">Maximum bytes to pin (default: 1GB)</param>
        public PinnedMemoryManager(int maxPinnedBytes = 1024 * 1024 * 1024)
        {
            _maxPinnedBytes = maxPinnedBytes;
            _allocations = new Dictionary<IntPtr, int>();
            _lock = new object();
        }

        /// <summary>
        /// Pin memory for a tensor
        /// </summary>
        /// <returns>Handle to pinned memory</returns>
        public PinnedMemoryHandle PinMemory(Tensor tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            var size = GetTensorSize(tensor);

            lock (_lock)
            {
                if (_totalPinnedBytes + size > _maxPinnedBytes)
                {
                    // Try to free some memory
                    if (!TryFreeMemory(size))
                    {
                        throw new CommunicationException("Cannot pin memory: limit exceeded");
                    }
                }

                // Get pointer to tensor data
                GCHandle handle = GCHandle.Alloc(tensor.Data, GCHandleType.Pinned);
                var ptr = GCHandle.ToIntPtr(handle);
                _allocations[ptr] = size;
                _totalPinnedBytes += size;

                return new PinnedMemoryHandle(ptr, size, this, handle);
            }
        }

        /// <summary>
        /// Unpin memory
        /// </summary>
        internal void UnpinMemory(IntPtr ptr, GCHandle handle)
        {
            lock (_lock)
            {
                if (_allocations.TryGetValue(ptr, out int size))
                {
                    handle.Free();
                    _allocations.Remove(ptr);
                    _totalPinnedBytes -= size;
                }
            }
        }

        /// <summary>
        /// Try to free memory to make space
        /// </summary>
        private bool TryFreeMemory(int requiredBytes)
        {
            // Simple LRU: free oldest allocations
            foreach (var kvp in _allocations.ToList())
            {
                if (_totalPinnedBytes + requiredBytes <= _maxPinnedBytes)
                {
                    break;
                }

                // Free the handle
                var handle = GCHandle.FromIntPtr(kvp.Key);
                handle.Free();
                _allocations.Remove(kvp.Key);
                _totalPinnedBytes -= kvp.Value;
            }

            return _totalPinnedBytes + requiredBytes <= _maxPinnedBytes;
        }

        private int GetTensorSize(Tensor tensor)
        {
            return tensor.Size * Marshal.SizeOf<float>();
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                lock (_lock)
                {
                    foreach (var ptr in _allocations.Keys.ToList())
                    {
                        var handle = GCHandle.FromIntPtr(ptr);
                        handle.Free();
                    }
                    _allocations.Clear();
                }
                _disposed = true;
            }
        }
    }
}
