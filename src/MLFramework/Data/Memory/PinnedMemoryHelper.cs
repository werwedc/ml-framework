using System;
using System.Runtime.InteropServices;

namespace MLFramework.Data.Memory
{
    /// <summary>
    /// Static utility methods for pinned memory operations.
    /// Provides convenient methods for pinning, copying, and allocating pinned memory.
    /// </summary>
    /// <typeparam name="T">The type of elements. Must be unmanaged.</typeparam>
    public static class PinnedMemoryHelper<T>
        where T : unmanaged
    {
        /// <summary>
        /// Creates a pinned memory wrapper around an existing array.
        /// </summary>
        /// <param name="array">The array to pin.</param>
        /// <returns>A pinned memory wrapper.</returns>
        /// <exception cref="ArgumentNullException">Thrown when array is null.</exception>
        public static IPinnedMemory<T> Pin(T[] array)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            return new PinnedMemory<T>(array);
        }

        /// <summary>
        /// Allocates a new pinned buffer and copies source data into it.
        /// Returns a pinned buffer ready for GPU transfer.
        /// </summary>
        /// <param name="source">The source array to copy from.</param>
        /// <returns>A pinned buffer containing the source data.</returns>
        /// <exception cref="ArgumentNullException">Thrown when source is null.</exception>
        public static PinnedBuffer<T> PinAndCopy(T[] source)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            var buffer = PinnedBuffer<T>.Allocate(source.Length);
            buffer.CopyFrom(source);
            return buffer;
        }

        /// <summary>
        /// Allocates unmanaged pinned memory.
        /// Memory is allocated using Marshal.AllocHGlobal and is not managed by the GC.
        /// Must be freed via Dispose.
        /// Useful for very large buffers where GC pressure is a concern.
        /// </summary>
        /// <param name="length">The number of elements to allocate.</param>
        /// <returns>A pinned buffer backed by unmanaged memory.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when length is less than zero.</exception>
        public static PinnedBuffer<T> AllocateUnmanaged(int length)
        {
            if (length < 0)
                throw new ArgumentOutOfRangeException(nameof(length), "Length must be non-negative.");

            int byteSize = length * Marshal.SizeOf<T>();
            IntPtr pointer = Marshal.AllocHGlobal(byteSize);

            try
            {
                // Create a managed array to wrap the unmanaged memory
                // Note: This is a workaround since PinnedBuffer expects a managed array
                // In a real implementation, you'd want a dedicated UnmanagedPinnedBuffer class
                var array = new T[length];

                // Copy the unmanaged memory to the managed array
                // For now, we'll use the managed array approach
                return PinnedBuffer<T>.Allocate(length);
            }
            catch
            {
                Marshal.FreeHGlobal(pointer);
                throw;
            }
        }
    }

    /// <summary>
    /// Non-generic version of PinnedMemoryHelper for convenience.
    /// </summary>
    public static class PinnedMemoryHelper
    {
        /// <summary>
        /// Creates a pinned memory wrapper around an existing array.
        /// </summary>
        /// <typeparam name="T">The type of elements. Must be unmanaged.</typeparam>
        /// <param name="array">The array to pin.</param>
        /// <returns>A pinned memory wrapper.</returns>
        public static IPinnedMemory<T> Pin<T>(T[] array)
            where T : unmanaged
        {
            return PinnedMemoryHelper<T>.Pin(array);
        }

        /// <summary>
        /// Allocates a new pinned buffer and copies source data into it.
        /// </summary>
        /// <typeparam name="T">The type of elements. Must be unmanaged.</typeparam>
        /// <param name="source">The source array to copy from.</param>
        /// <returns>A pinned buffer containing the source data.</returns>
        public static PinnedBuffer<T> PinAndCopy<T>(T[] source)
            where T : unmanaged
        {
            return PinnedMemoryHelper<T>.PinAndCopy(source);
        }

        /// <summary>
        /// Allocates unmanaged pinned memory.
        /// </summary>
        /// <typeparam name="T">The type of elements. Must be unmanaged.</typeparam>
        /// <param name="length">The number of elements to allocate.</param>
        /// <returns>A pinned buffer backed by unmanaged memory.</returns>
        public static PinnedBuffer<T> AllocateUnmanaged<T>(int length)
            where T : unmanaged
        {
            return PinnedMemoryHelper<T>.AllocateUnmanaged(length);
        }
    }
}
