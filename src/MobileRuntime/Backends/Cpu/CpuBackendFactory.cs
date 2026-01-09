using MLFramework.MobileRuntime.Backends.Cpu.Interfaces;
using MLFramework.MobileRuntime.Backends.Cpu.Models;
using MLFramework.MobileRuntime.Memory;

namespace MLFramework.MobileRuntime.Backends.Cpu
{
    using System.Runtime.InteropServices;

    /// <summary>
    /// Factory for creating CPU backend instances.
    /// </summary>
    public static class CpuBackendFactory
    {
        /// <summary>
        /// Creates a default CPU backend with auto-detected capabilities.
        /// </summary>
        /// <param name="memoryPool">Memory pool for tensor allocations.</param>
        /// <param name="tensorFactory">Tensor factory for creating tensors.</param>
        /// <returns>A configured CPU backend instance.</returns>
        public static ICpuBackend CreateDefault(IMemoryPool memoryPool, ITensorFactory tensorFactory)
        {
            if (memoryPool == null)
                throw new System.ArgumentNullException(nameof(memoryPool));
            if (tensorFactory == null)
                throw new System.ArgumentNullException(nameof(tensorFactory));

            return new CpuBackend(memoryPool, tensorFactory);
        }

        /// <summary>
        /// Creates a CPU backend with NEON optimizations enabled for ARM platforms.
        /// </summary>
        /// <param name="memoryPool">Memory pool for tensor allocations.</param>
        /// <param name="tensorFactory">Tensor factory for creating tensors.</param>
        /// <returns>A CPU backend optimized for ARM NEON.</returns>
        public static ICpuBackend CreateWithNeonOptimization(IMemoryPool memoryPool, ITensorFactory tensorFactory)
        {
            if (memoryPool == null)
                throw new System.ArgumentNullException(nameof(memoryPool));
            if (tensorFactory == null)
                throw new System.ArgumentNullException(nameof(tensorFactory));

            var backend = new CpuBackend(memoryPool, tensorFactory);

            // On ARM64, NEON should be automatically detected
            var cpuInfo = backend.GetCpuInfo();

            if (cpuInfo.Vendor == "ARM" && cpuInfo.Capabilities.SupportsNeon)
            {
                // Vectorization is already enabled by default
                // This factory method is for clarity and explicit configuration
            }

            return backend;
        }

        /// <summary>
        /// Creates a CPU backend optimized for x86/x64 platforms.
        /// </summary>
        /// <param name="memoryPool">Memory pool for tensor allocations.</param>
        /// <param name="tensorFactory">Tensor factory for creating tensors.</param>
        /// <returns>A CPU backend optimized for x86.</returns>
        public static ICpuBackend CreateForX86(IMemoryPool memoryPool, ITensorFactory tensorFactory)
        {
            if (memoryPool == null)
                throw new System.ArgumentNullException(nameof(memoryPool));
            if (tensorFactory == null)
                throw new System.ArgumentNullException(nameof(tensorFactory));

            var backend = new CpuBackend(memoryPool, tensorFactory);

            var cpuInfo = backend.GetCpuInfo();

            if (cpuInfo.Vendor == "x86/x64")
            {
                // Ensure vectorization is enabled for x86
                backend.EnableVectorization(true);
            }

            return backend;
        }

        /// <summary>
        /// Creates a CPU backend with custom configuration.
        /// </summary>
        /// <param name="memoryPool">Memory pool for tensor allocations.</param>
        /// <param name="tensorFactory">Tensor factory for creating tensors.</param>
        /// <param name="enableVectorization">Whether to enable SIMD vectorization.</param>
        /// <param name="enableMultiThreading">Whether to enable multi-threading.</param>
        /// <param name="maxThreads">Maximum number of threads (0 = auto-detect).</param>
        /// <returns>A configured CPU backend instance.</returns>
        public static ICpuBackend CreateCustom(
            IMemoryPool memoryPool,
            ITensorFactory tensorFactory,
            bool enableVectorization,
            bool enableMultiThreading,
            int maxThreads = 0)
        {
            if (memoryPool == null)
                throw new System.ArgumentNullException(nameof(memoryPool));
            if (tensorFactory == null)
                throw new System.ArgumentNullException(nameof(tensorFactory));

            var backend = new CpuBackend(memoryPool, tensorFactory);

            backend.EnableVectorization(enableVectorization);
            backend.EnableMultiThreading(enableMultiThreading, maxThreads);

            return backend;
        }
    }
}
