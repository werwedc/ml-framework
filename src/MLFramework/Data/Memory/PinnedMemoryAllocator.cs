using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;

namespace MLFramework.Data.Memory
{
    /// <summary>
    /// Implementation of pinned memory allocator for faster GPU data transfers via DMA.
    /// Uses Marshal.AllocHGlobal as a placeholder for CUDA pinned memory (cudaMallocHost).
    /// </summary>
    public class PinnedMemoryAllocator : IPinnedMemoryAllocator
    {
        private readonly bool _pinnedMemorySupported;
        private readonly Dictionary<IntPtr, int> _allocatedBlocks;
        private readonly object _lock;
        private volatile bool _isDisposed;

        /// <summary>
        /// Initializes a new instance of the PinnedMemoryAllocator class.
        /// </summary>
        /// <param name="forcePinned">If true, forces pinned memory support regardless of CUDA detection.</param>
        public PinnedMemoryAllocator(bool forcePinned = false)
        {
            _lock = new object();
            _allocatedBlocks = new Dictionary<IntPtr, int>();
            _isDisposed = false;

            // Detect CUDA support (placeholder - integrate with CUDA library later)
            _pinnedMemorySupported = forcePinned || DetectCudaSupport();
        }

        /// <summary>
        /// Detects CUDA support for pinned memory allocation.
        /// Placeholder implementation - will integrate with CUDA library in the future.
        /// </summary>
        /// <returns>True if CUDA is available, false otherwise.</returns>
        private bool DetectCudaSupport()
        {
            // Placeholder - will integrate with CUDA library
            // For now, assume CUDA is available
            return true;
        }

        /// <summary>
        /// Allocates pinned memory of the specified size.
        /// </summary>
        /// <param name="size">Size in bytes to allocate.</param>
        /// <returns>Pointer to the allocated memory.</returns>
        /// <exception cref="ObjectDisposedException">Thrown when the allocator is disposed.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when size is less than or equal to zero.</exception>
        public IntPtr Allocate(int size)
        {
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(PinnedMemoryAllocator));

            if (size <= 0)
                throw new ArgumentOutOfRangeException(nameof(size), "Size must be greater than zero.");

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

        /// <summary>
        /// Allocates pinned memory using CUDA API.
        /// Placeholder for CUDA cudaMallocHost / cudaHostAlloc.
        /// </summary>
        /// <param name="size">Size in bytes to allocate.</param>
        /// <returns>Pointer to the allocated pinned memory.</returns>
        private IntPtr AllocatePinnedMemory(int size)
        {
            // Placeholder for CUDA cudaMallocHost / cudaHostAlloc
            // For now, use AllocHGlobal as fallback
            return Marshal.AllocHGlobal(size);
        }

        /// <summary>
        /// Frees previously allocated pinned memory.
        /// </summary>
        /// <param name="pointer">Pointer to the memory to free.</param>
        /// <exception cref="ObjectDisposedException">Thrown when the allocator is disposed.</exception>
        /// <exception cref="ArgumentException">Thrown when pointer is zero or was not allocated by this allocator.</exception>
        public void Free(IntPtr pointer)
        {
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(PinnedMemoryAllocator));

            if (pointer == IntPtr.Zero)
                throw new ArgumentException("Pointer cannot be zero.", nameof(pointer));

            int size;

            lock (_lock)
            {
                if (!_allocatedBlocks.TryGetValue(pointer, out size))
                    throw new ArgumentException("Pointer was not allocated by this allocator.", nameof(pointer));

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

        /// <summary>
        /// Frees pinned memory using CUDA API.
        /// Placeholder for CUDA cudaFreeHost.
        /// </summary>
        /// <param name="pointer">Pointer to the pinned memory to free.</param>
        private void FreePinnedMemory(IntPtr pointer)
        {
            // Placeholder for CUDA cudaFreeHost
            // For now, use FreeHGlobal
            Marshal.FreeHGlobal(pointer);
        }

        /// <summary>
        /// Gets whether pinned memory is supported on the current platform.
        /// </summary>
        public bool IsPinnedMemorySupported => _pinnedMemorySupported;

        /// <summary>
        /// Copies data from a byte array to pinned memory.
        /// </summary>
        /// <param name="pinnedPtr">Pointer to the pinned memory.</param>
        /// <param name="data">Byte array containing the data to copy.</param>
        /// <param name="offset">Offset in the data array to start copying from.</param>
        /// <param name="length">Number of bytes to copy. If null, copies all remaining bytes from offset.</param>
        /// <exception cref="ArgumentException">Thrown when pinnedPtr is zero or offset/length are out of range.</exception>
        /// <exception cref="ArgumentNullException">Thrown when data is null.</exception>
        public static void CopyToPinnedMemory(IntPtr pinnedPtr, byte[] data, int offset = 0, int? length = null)
        {
            if (pinnedPtr == IntPtr.Zero)
                throw new ArgumentException("Pointer cannot be zero.", nameof(pinnedPtr));

            if (data == null)
                throw new ArgumentNullException(nameof(data));

            int copyLength = length ?? data.Length;

            if (offset < 0 || offset >= data.Length)
                throw new ArgumentOutOfRangeException(nameof(offset), "Offset is out of range.");

            if (copyLength < 0 || offset + copyLength > data.Length)
                throw new ArgumentOutOfRangeException(nameof(length), "Length is out of range.");

            Marshal.Copy(data, offset, pinnedPtr, copyLength);
        }

        /// <summary>
        /// Copies data from pinned memory to a byte array.
        /// </summary>
        /// <param name="pinnedPtr">Pointer to the pinned memory.</param>
        /// <param name="data">Byte array to copy data into.</param>
        /// <param name="offset">Offset in the data array to start copying to.</param>
        /// <param name="length">Number of bytes to copy.</param>
        /// <exception cref="ArgumentException">Thrown when pinnedPtr is zero or offset/length are out of range.</exception>
        /// <exception cref="ArgumentNullException">Thrown when data is null.</exception>
        public static void CopyFromPinnedMemory(IntPtr pinnedPtr, byte[] data, int offset = 0, int length = 0)
        {
            if (pinnedPtr == IntPtr.Zero)
                throw new ArgumentException("Pointer cannot be zero.", nameof(pinnedPtr));

            if (data == null)
                throw new ArgumentNullException(nameof(data));

            if (length == 0)
                length = data.Length;

            if (offset < 0 || offset >= data.Length)
                throw new ArgumentOutOfRangeException(nameof(offset), "Offset is out of range.");

            if (length < 0 || offset + length > data.Length)
                throw new ArgumentOutOfRangeException(nameof(length), "Length is out of range.");

            Marshal.Copy(pinnedPtr, data, offset, length);
        }

        /// <summary>
        /// Disposes the allocator and frees all allocated memory.
        /// </summary>
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

            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Finalizer to ensure memory is freed.
        /// </summary>
        ~PinnedMemoryAllocator()
        {
            Dispose();
        }
    }
}
