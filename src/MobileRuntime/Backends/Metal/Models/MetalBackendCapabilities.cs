namespace MobileRuntime.Backends.Metal
{
    /// <summary>
    /// Capabilities of the Metal backend
    /// </summary>
    public class MetalBackendCapabilities
    {
        /// <summary>
        /// Gets or sets whether MPS (Metal Performance Shaders) is supported
        /// </summary>
        public bool SupportsMPS { get; set; }

        /// <summary>
        /// Gets or sets whether unified memory is supported
        /// </summary>
        public bool SupportsUnifiedMemory { get; set; }

        /// <summary>
        /// Gets or sets the maximum number of threads per threadgroup
        /// </summary>
        public int MaxThreadsPerThreadgroup { get; set; }

        /// <summary>
        /// Gets or sets the maximum texture width
        /// </summary>
        public int MaxTextureWidth { get; set; }

        /// <summary>
        /// Gets or sets the maximum texture height
        /// </summary>
        public int MaxTextureHeight { get; set; }

        /// <summary>
        /// Gets or sets the maximum buffer length
        /// </summary>
        public int MaxBufferLength { get; set; }
    }
}
