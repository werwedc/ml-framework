using System.Runtime.InteropServices;

namespace MLFramework.MobileRuntime.Memory
{
    /// <summary>
    /// Factory for creating memory pools with different configurations.
    /// </summary>
    public static class MemoryPoolFactory
    {
        /// <summary>
        /// Creates a default memory pool with the specified capacity.
        /// </summary>
        /// <param name="capacity">Initial capacity in bytes (default: 16MB).</param>
        /// <returns>A new DefaultMemoryPool instance.</returns>
        public static IMemoryPool CreateDefault(long capacity = 16 * 1024 * 1024)
        {
            return new DefaultMemoryPool(capacity);
        }

        /// <summary>
        /// Creates a pre-allocated memory pool with the specified total size.
        /// </summary>
        /// <param name="totalSize">Total size of the memory pool in bytes.</param>
        /// <returns>A new PreallocatedMemoryPool instance.</returns>
        public static IMemoryPool CreatePreallocated(long totalSize)
        {
            return new PreallocatedMemoryPool(totalSize);
        }

        /// <summary>
        /// Creates a memory pool configured for low memory mode.
        /// </summary>
        /// <returns>A memory pool optimized for low memory environments.</returns>
        public static IMemoryPool CreateLowMemoryMode()
        {
            var pool = new DefaultMemoryPool(4 * 1024 * 1024); // 4MB default
            pool.EnableLowMemoryMode(true);
            return pool;
        }

        /// <summary>
        /// Creates an optimal memory pool based on the current platform capabilities.
        /// </summary>
        /// <returns>A memory pool configured for the current platform.</returns>
        public static IMemoryPool CreateOptimalForPlatform()
        {
            if (IsMobilePlatform())
            {
                // Mobile platforms use smaller pools
                return CreateDefault(8 * 1024 * 1024); // 8MB
            }
            else
            {
                // Desktop platforms use larger pools
                return CreateDefault(32 * 1024 * 1024); // 32MB
            }
        }

        /// <summary>
        /// Creates a memory pool configured for specific model types.
        /// </summary>
        /// <param name="modelType">The type of model (e.g., MNIST, CIFAR, ImageNet).</param>
        /// <returns>A memory pool configured for the specified model type.</returns>
        public static IMemoryPool CreateForModel(ModelType modelType)
        {
            long poolSize = modelType switch
            {
                ModelType.MNIST => 8 * 1024 * 1024,      // 8MB
                ModelType.CIFAR => 16 * 1024 * 1024,     // 16MB
                ModelType.ImageNet => 64 * 1024 * 1024,  // 64MB
                ModelType.Custom => 32 * 1024 * 1024,    // 32MB
                _ => 16 * 1024 * 1024                    // Default: 16MB
            };

            return CreateDefault(poolSize);
        }

        private static bool IsMobilePlatform()
        {
            // Android and iOS are not available in .NET Standard 2.0
            // Use runtime detection based on available platforms
            var os = Environment.OSVersion;
            var platform = os.Platform;

            // Assume it's mobile if not Windows/Linux/macOS
            // In practice, you'd use runtime-specific detection or compile-time directives
            return platform != PlatformID.Win32NT &&
                   platform != PlatformID.Unix &&
                   platform != PlatformID.MacOSX;
        }

        /// <summary>
        /// Model types for memory pool configuration.
        /// </summary>
        public enum ModelType
        {
            /// <summary>
            /// Small model (e.g., MNIST).
            /// </summary>
            MNIST,

            /// <summary>
            /// Medium model (e.g., CIFAR-10/100).
            /// </summary>
            CIFAR,

            /// <summary>
            /// Large model (e.g., ImageNet models).
            /// </summary>
            ImageNet,

            /// <summary>
            /// Custom model size.
            /// </summary>
            Custom
        }
    }
}
