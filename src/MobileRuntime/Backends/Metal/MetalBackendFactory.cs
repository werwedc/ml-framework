using System;

namespace MobileRuntime.Backends.Metal
{
    /// <summary>
    /// Factory for creating Metal backend instances
    /// </summary>
    public static class MetalBackendFactory
    {
        /// <summary>
        /// Creates a default Metal backend
        /// </summary>
        public static IMetalBackend CreateDefault(ITensorFactory tensorFactory)
        {
            if (!IsAvailable())
                throw new InvalidOperationException("Metal is not available on this device");

            return new MetalBackend(tensorFactory);
        }

        /// <summary>
        /// Creates a Metal backend with MPS (Metal Performance Shaders) fallback
        /// </summary>
        public static IMetalBackend CreateWithMPSFallback(ITensorFactory tensorFactory)
        {
            var backend = CreateDefault(tensorFactory);

            // Check if MPS is available
            if (backend.Capabilities.SupportsMPS)
            {
                // Use MPS-accelerated operations when available
                // This would be implemented by using MPSGraph for supported operations
            }

            return backend;
        }

        /// <summary>
        /// Checks if Metal is available on the current device
        /// </summary>
        public static bool IsAvailable()
        {
            try
            {
                // Try to create a Metal device
                // This would call MTLCreateSystemDefaultDevice() and check if it returns null
                return true; // Placeholder - actual implementation would check availability
            }
            catch
            {
                return false;
            }
        }
    }
}
