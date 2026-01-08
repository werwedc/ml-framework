using System;
using System.Buffers;
using System.Runtime.InteropServices;

namespace MLFramework.Data.Memory
{
    /// <summary>
    /// Pinned memory with automatic cleanup and pooling integration.
    /// Provides utility methods for copying and manipulating pinned data.
    /// </summary>
    /// <typeparam name="T">The type of elements in the buffer. Must be unmanaged.</typeparam>
    public sealed class PinnedBuffer<T> : IPinnedMemory<T>
        where T : unmanaged
    {
        private readonly T[] _array;
        private GCHandle _gcHandle;
        private readonly int _length;
        private bool _isDisposed;

        /// <summary>
        /// Initializes a new instance of the PinnedBuffer class with an existing array.
        /// </summary>
        /// <param name="array">The array to pin. Must not be null.</param>
        /// <exception cref="ArgumentNullException">Thrown when array is null.</exception>
        private PinnedBuffer(T[] array)
        {
            _array = array ?? throw new ArgumentNullException(nameof(array));
            _length = array.Length;

            // Pin the array immediately
            _gcHandle = GCHandle.Alloc(array, GCHandleType.Pinned);
        }

        /// <summary>
        /// Allocates a new pinned buffer of the specified length.
        /// </summary>
        /// <param name="length">The length of the buffer to allocate.</param>
        /// <returns>A new pinned buffer ready for GPU transfer.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when length is less than zero.</exception>
        public static PinnedBuffer<T> Allocate(int length)
        {
            if (length < 0)
                throw new ArgumentOutOfRangeException(nameof(length), "Length must be non-negative.");

            // Allocate new array and pin it
            var array = new T[length];
            return new PinnedBuffer<T>(array);
        }

        /// <summary>
        /// Gets the underlying array (read-only).
        /// </summary>
        public T[] Array => _array;

        /// <summary>
        /// Gets a safe span access to the underlying array data.
        /// </summary>
        public Span<T> Span => _array.AsSpan();

        /// <summary>
        /// Gets the pointer to the pinned memory.
        /// </summary>
        public IntPtr Pointer => _gcHandle.AddrOfPinnedObject();

        /// <summary>
        /// Gets the length of the buffer.
        /// </summary>
        public int Length => _length;

        /// <summary>
        /// Gets whether the memory is currently pinned.
        /// </summary>
        public bool IsPinned => _gcHandle.IsAllocated;

        /// <summary>
        /// Copies data from a source array to the pinned buffer.
        /// </summary>
        /// <param name="source">The source array to copy from.</param>
        /// <param name="sourceOffset">The offset in the source array to start copying from.</param>
        /// <exception cref="ArgumentNullException">Thrown when source is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when offsets or lengths are invalid.</exception>
        public void CopyFrom(T[] source, int sourceOffset = 0)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            if (sourceOffset < 0 || sourceOffset >= source.Length)
                throw new ArgumentOutOfRangeException(nameof(sourceOffset), "Source offset is out of range.");

            if (source.Length - sourceOffset < _length)
                throw new ArgumentOutOfRangeException(nameof(source), "Source array is too short.");

            System.Array.Copy(source, sourceOffset, _array, 0, _length);
        }

        /// <summary>
        /// Copies data from a source span to the pinned buffer.
        /// </summary>
        /// <param name="source">The source span to copy from.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when source span length doesn't match buffer length.</exception>
        public void CopyFrom(Span<T> source)
        {
            if (source.Length != _length)
                throw new ArgumentOutOfRangeException(nameof(source), "Source span length must match buffer length.");

            source.CopyTo(_array);
        }

        /// <summary>
        /// Copies data from the pinned buffer to a destination array.
        /// </summary>
        /// <param name="destination">The destination array to copy to.</param>
        /// <param name="destinationOffset">The offset in the destination array to start copying to.</param>
        /// <exception cref="ArgumentNullException">Thrown when destination is null.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when offsets or lengths are invalid.</exception>
        public void CopyTo(T[] destination, int destinationOffset = 0)
        {
            if (destination == null)
                throw new ArgumentNullException(nameof(destination));

            if (destinationOffset < 0 || destinationOffset >= destination.Length)
                throw new ArgumentOutOfRangeException(nameof(destinationOffset), "Destination offset is out of range.");

            if (destination.Length - destinationOffset < _length)
                throw new ArgumentOutOfRangeException(nameof(destination), "Destination array is too short.");

            System.Array.Copy(_array, 0, destination, destinationOffset, _length);
        }

        /// <summary>
        /// Copies data from the pinned buffer to a destination span.
        /// </summary>
        /// <param name="destination">The destination span to copy to.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when destination span length doesn't match buffer length.</exception>
        public void CopyTo(Span<T> destination)
        {
            if (destination.Length != _length)
                throw new ArgumentOutOfRangeException(nameof(destination), "Destination span length must match buffer length.");

            _array.AsSpan().CopyTo(destination);
        }

        /// <summary>
        /// Sets all elements in the buffer to the specified value.
        /// Useful for zeroing buffers.
        /// </summary>
        /// <param name="value">The value to fill the buffer with.</param>
        public void Fill(T value)
        {
            _array.AsSpan().Fill(value);
        }

        /// <summary>
        /// Unpins the memory, allowing the garbage collector to move the underlying array.
        /// </summary>
        public void Unpin()
        {
            if (_gcHandle.IsAllocated)
            {
                _gcHandle.Free();
            }
        }

        /// <summary>
        /// Disposes the pinned buffer and unpins the array.
        /// Implements the IDisposable pattern.
        /// </summary>
        public void Dispose()
        {
            if (!_isDisposed)
            {
                Unpin();
                _isDisposed = true;
            }
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Finalizer to ensure pinned memory is released.
        /// </summary>
        ~PinnedBuffer()
        {
            Dispose();
        }
    }
}
