using System;

namespace MLFramework.MobileRuntime.Memory
{
    /// <summary>
    /// Interface for managing tensor memory pools to reduce allocation overhead.
    /// </summary>
    public interface IMemoryPool : IDisposable
    {
        /// <summary>
        /// Allocates memory from the pool for a tensor.
        /// </summary>
        /// <param name="size">Size in bytes to allocate.</param>
        /// <param name="dataType">Data type of the tensor (used for size calculation hints).</param>
        /// <returns>Pointer to the allocated memory.</returns>
        IntPtr Allocate(long size, DataType dataType);

        /// <summary>
        /// Frees memory back to the pool.
        /// </summary>
        /// <param name="ptr">Pointer to the memory to free.</param>
        /// <param name="size">Size in bytes of the memory being freed.</param>
        void Free(IntPtr ptr, long size);

        /// <summary>
        /// Sets the maximum memory limit for the pool.
        /// </summary>
        /// <param name="maxBytes">Maximum memory limit in bytes.</param>
        void SetMemoryLimit(long maxBytes);

        /// <summary>
        /// Gets the available memory in the pool.
        /// </summary>
        /// <returns>Available memory in bytes.</returns>
        long GetAvailableMemory();

        /// <summary>
        /// Gets the currently used memory in the pool.
        /// </summary>
        /// <returns>Used memory in bytes.</returns>
        long GetUsedMemory();

        /// <summary>
        /// Gets statistics about the memory pool usage.
        /// </summary>
        /// <returns>Memory pool statistics.</returns>
        MemoryPoolStats GetStats();

        /// <summary>
        /// Enables or disables low memory mode.
        /// In low memory mode, blocks are not cached and released immediately.
        /// </summary>
        /// <param name="enable">True to enable low memory mode, false to disable.</param>
        void EnableLowMemoryMode(bool enable);

        /// <summary>
        /// Pre-allocates memory for tensors of a specific size.
        /// </summary>
        /// <param name="size">Size in bytes to pre-allocate.</param>
        void PreAllocateForTensor(long size);

        /// <summary>
        /// Resets the memory pool, freeing all allocated memory.
        /// </summary>
        void Reset();

        /// <summary>
        /// Disposes the memory pool, releasing all allocated resources.
        /// </summary>
        new void Dispose();
    }
}
