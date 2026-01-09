using System;

namespace MobileRuntime.Interfaces
{
    /// <summary>
    /// Interface for memory pool to manage tensor memory allocations
    /// </summary>
    public interface IMemoryPool : IDisposable
    {
        /// <summary>
        /// Allocates memory for a tensor
        /// </summary>
        /// <param name="size">Size in bytes</param>
        /// <param name="dataType">Data type</param>
        /// <returns>Pointer to allocated memory</returns>
        IntPtr Allocate(long size, DataType dataType);

        /// <summary>
        /// Frees previously allocated memory
        /// </summary>
        /// <param name="ptr">Pointer to free</param>
        /// <param name="size">Size in bytes</param>
        void Free(IntPtr ptr, long size);

        /// <summary>
        /// Gets memory pool statistics
        /// </summary>
        MemoryPoolStats GetStats();

        /// <summary>
        /// Sets memory limit
        /// </summary>
        /// <param name="limitInBytes">Memory limit in bytes</param>
        void SetMemoryLimit(long limitInBytes);
    }

    /// <summary>
    /// Memory pool statistics
    /// </summary>
    public class MemoryPoolStats
    {
        public long TotalAllocatedBytes { get; set; }
        public long PeakUsage { get; set; }
        public int CacheHits { get; set; }
        public int CacheMisses { get; set; }
        public long MemoryLimit { get; set; }
        public int ActiveAllocations { get; set; }
    }
}
