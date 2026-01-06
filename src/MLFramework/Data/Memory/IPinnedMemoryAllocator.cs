using System;

namespace MLFramework.Data.Memory
{
    /// <summary>
    /// Interface for pinned memory allocation to enable faster GPU data transfers via DMA.
    /// </summary>
    public interface IPinnedMemoryAllocator : IDisposable
    {
        /// <summary>
        /// Allocates pinned memory of the specified size.
        /// </summary>
        /// <param name="size">Size in bytes to allocate.</param>
        /// <returns>Pointer to the allocated memory.</returns>
        IntPtr Allocate(int size);

        /// <summary>
        /// Frees previously allocated pinned memory.
        /// </summary>
        /// <param name="pointer">Pointer to the memory to free.</param>
        void Free(IntPtr pointer);

        /// <summary>
        /// Gets whether pinned memory is supported on the current platform.
        /// </summary>
        bool IsPinnedMemorySupported { get; }
    }
}
