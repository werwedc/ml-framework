using System;

namespace MLFramework.Data.Memory
{
    /// <summary>
    /// Memory pinning strategies for optimal CPU-to-GPU data transfers.
    /// Different strategies are suitable for different buffer sizes and usage patterns.
    /// </summary>
    public enum PinningStrategy
    {
        /// <summary>
        /// Don't pin. Falls back to regular copy with intermediate buffering.
        /// Suitable for very small buffers (< 1KB) where pinning overhead outweighs benefits.
        /// </summary>
        None = 0,

        /// <summary>
        /// Use GCHandle for pinning. Works for managed arrays and is the default strategy.
        /// Suitable for medium buffers (< 1MB) with moderate lifetimes.
        /// </summary>
        GCHandle = 1,

        /// <summary>
        /// Use unmanaged allocation (Marshal.AllocHGlobal).
        /// Memory is not managed by GC, which reduces GC pressure.
        /// Suitable for large buffers (>= 1MB) or long-lived buffers.
        /// </summary>
        Unmanaged = 2,

        /// <summary>
        /// Use pooled pinned memory (PinnedMemoryPool).
        /// Reuses pinned buffers to reduce allocation overhead.
        /// Suitable for scenarios with high reuse and frequent allocation/deallocation.
        /// </summary>
        PinnedObjectPool = 3
    }

    /// <summary>
    /// Strategy selector for choosing the optimal pinning method based on buffer characteristics.
    /// </summary>
    public static class PinningStrategySelector
    {
        private const int SmallBufferSizeThreshold = 1024; // 1KB in bytes
        private const int LargeBufferSizeThreshold = 1024 * 1024; // 1MB in bytes
        private const int HighReuseThreshold = 10; // Number of expected reuses

        /// <summary>
        /// Selects the optimal pinning strategy based on buffer size and expected usage pattern.
        /// </summary>
        /// <typeparam name="T">The type of elements in the buffer.</typeparam>
        /// <param name="bufferSize">Size of the buffer in elements.</param>
        /// <param name="expectedLifetime">Expected number of uses or lifetime of the buffer.</param>
        /// <returns>The optimal pinning strategy.</returns>
        public static PinningStrategy SelectStrategy<T>(int bufferSize, int expectedLifetime)
            where T : unmanaged
        {
            if (bufferSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(bufferSize), "Buffer size must be greater than zero.");

            int byteSize = bufferSize * System.Runtime.InteropServices.Marshal.SizeOf<T>();

            return SelectStrategyByByteSize(byteSize, expectedLifetime);
        }

        /// <summary>
        /// Selects the optimal pinning strategy based on buffer size in bytes and expected usage pattern.
        /// </summary>
        /// <param name="byteSize">Size of the buffer in bytes.</param>
        /// <param name="expectedLifetime">Expected number of uses or lifetime of the buffer.</param>
        /// <returns>The optimal pinning strategy.</returns>
        public static PinningStrategy SelectStrategyByByteSize(int byteSize, int expectedLifetime)
        {
            if (byteSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(byteSize), "Byte size must be greater than zero.");

            if (expectedLifetime < 0)
                throw new ArgumentOutOfRangeException(nameof(expectedLifetime), "Expected lifetime must be non-negative.");

            // Very small buffers: no pinning
            if (byteSize < SmallBufferSizeThreshold)
            {
                return PinningStrategy.None;
            }

            // Large buffers with high reuse: use pool
            if (byteSize >= LargeBufferSizeThreshold && expectedLifetime >= HighReuseThreshold)
            {
                return PinningStrategy.PinnedObjectPool;
            }

            // Large buffers: use unmanaged allocation
            if (byteSize >= LargeBufferSizeThreshold)
            {
                return PinningStrategy.Unmanaged;
            }

            // Medium buffers with high reuse: use pool
            if (expectedLifetime >= HighReuseThreshold)
            {
                return PinningStrategy.PinnedObjectPool;
            }

            // Default: use GCHandle
            return PinningStrategy.GCHandle;
        }

        /// <summary>
        /// Gets the threshold for small buffers (in bytes).
        /// Buffers smaller than this threshold use None strategy.
        /// </summary>
        public static int SmallBufferSizeThresholdBytes => SmallBufferSizeThreshold;

        /// <summary>
        /// Gets the threshold for large buffers (in bytes).
        /// Buffers larger than or equal to this threshold use Unmanaged or PinnedObjectPool strategy.
        /// </summary>
        public static int LargeBufferSizeThresholdBytes => LargeBufferSizeThreshold;

        /// <summary>
        /// Gets the threshold for high reuse.
        /// Expected lifetimes at or above this threshold favor pooling.
        /// </summary>
        public static int HighReuseThresholdCount => HighReuseThreshold;

        /// <summary>
        /// Creates a pinned memory object based on the selected strategy.
        /// </summary>
        /// <typeparam name="T">The type of elements in the buffer.</typeparam>
        /// <param name="array">The array to pin.</param>
        /// <param name="strategy">The pinning strategy to use.</param>
        /// <param name="pool">Optional pool to use for PinnedObjectPool strategy.</param>
        /// <returns>A pinned memory object, or null if strategy is None.</returns>
        public static IPinnedMemory<T> CreatePinnedMemory<T>(
            T[] array,
            PinningStrategy strategy,
            PinnedMemoryPool<T>? pool = null)
            where T : unmanaged
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            switch (strategy)
            {
                case PinningStrategy.None:
                    return null;

                case PinningStrategy.GCHandle:
                    return new PinnedMemory<T>(array);

                case PinningStrategy.Unmanaged:
                    // Create a buffer that copies to unmanaged memory
                    return PinnedMemoryHelper<T>.PinAndCopy(array);

                case PinningStrategy.PinnedObjectPool:
                    if (pool == null)
                        throw new ArgumentNullException(nameof(pool), "Pool must be provided for PinnedObjectPool strategy.");

                    if (pool.BufferSize != array.Length)
                        throw new ArgumentException("Pool buffer size doesn't match array length.", nameof(pool));

                    var buffer = pool.Rent();
                    buffer.CopyFrom(array);
                    return buffer;

                default:
                    throw new ArgumentException($"Unknown pinning strategy: {strategy}", nameof(strategy));
            }
        }
    }
}
