using System;

namespace MLFramework.Data.Memory
{
    /// <summary>
    /// Defines the contract for pinned memory buffers.
    /// Pinned memory prevents the garbage collector from moving data, enabling direct GPU access without intermediate copies.
    /// </summary>
    /// <typeparam name="T">The type of elements in the pinned memory. Must be unmanaged.</typeparam>
    public interface IPinnedMemory<T> : IDisposable
        where T : unmanaged
    {
        /// <summary>
        /// Gets a safe span access to the underlying array data.
        /// </summary>
        Span<T> Span { get; }

        /// <summary>
        /// Gets the pointer to the pinned memory.
        /// </summary>
        IntPtr Pointer { get; }

        /// <summary>
        /// Gets the length of the pinned memory buffer.
        /// </summary>
        int Length { get; }

        /// <summary>
        /// Gets whether the memory is currently pinned.
        /// </summary>
        bool IsPinned { get; }

        /// <summary>
        /// Unpins the memory, allowing the garbage collector to move the underlying array.
        /// </summary>
        void Unpin();
    }
}
