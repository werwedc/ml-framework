using System;
using System.Runtime.InteropServices;

namespace MLFramework.Data.Memory
{
    /// <summary>
    /// Basic pinned memory wrapper using GCHandle.
    /// Pins a managed array to prevent garbage collector from moving it.
    /// </summary>
    /// <typeparam name="T">The type of elements in the array. Must be unmanaged.</typeparam>
    public class PinnedMemory<T> : IPinnedMemory<T>
        where T : unmanaged
    {
        private readonly T[] _array;
        private GCHandle _gcHandle;
        private bool _isDisposed;

        /// <summary>
        /// Initializes a new instance of the PinnedMemory class.
        /// </summary>
        /// <param name="array">The array to pin. Must not be null.</param>
        /// <exception cref="ArgumentNullException">Thrown when array is null.</exception>
        public PinnedMemory(T[] array)
        {
            _array = array ?? throw new ArgumentNullException(nameof(array));

            // Pin the array to prevent GC from moving it
            _gcHandle = GCHandle.Alloc(array, GCHandleType.Pinned);
        }

        /// <summary>
        /// Gets a safe span access to the underlying array data.
        /// </summary>
        public Span<T> Span => _array.AsSpan();

        /// <summary>
        /// Gets the pointer to the pinned memory.
        /// </summary>
        public IntPtr Pointer => _gcHandle.AddrOfPinnedObject();

        /// <summary>
        /// Gets the length of the pinned memory buffer.
        /// </summary>
        public int Length => _array.Length;

        /// <summary>
        /// Gets whether the memory is currently pinned.
        /// </summary>
        public bool IsPinned => _gcHandle.IsAllocated;

        /// <summary>
        /// Unpins the memory, allowing the garbage collector to move the underlying array.
        /// Safe to call multiple times (no-op after first call).
        /// </summary>
        public void Unpin()
        {
            if (_gcHandle.IsAllocated)
            {
                _gcHandle.Free();
            }
        }

        /// <summary>
        /// Disposes the pinned memory wrapper and unpins the array.
        /// Implements the IDisposable pattern.
        /// Safe to call multiple times.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Protected dispose method for the dispose pattern.
        /// </summary>
        /// <param name="disposing">True if disposing managed resources.</param>
        protected virtual void Dispose(bool disposing)
        {
            if (!_isDisposed)
            {
                if (disposing)
                {
                    // Unpin the array
                    Unpin();
                }

                _isDisposed = true;
            }
        }

        /// <summary>
        /// Finalizer to ensure pinned memory is released.
        /// </summary>
        ~PinnedMemory()
        {
            Dispose(false);
        }
    }
}
