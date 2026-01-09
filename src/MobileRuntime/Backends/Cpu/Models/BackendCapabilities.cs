namespace MLFramework.MobileRuntime.Backends.Cpu.Models
{
    /// <summary>
    /// Represents the capabilities of the CPU backend.
    /// </summary>
    public class BackendCapabilities
    {
        /// <summary>
        /// Whether ARM NEON SIMD is supported.
        /// </summary>
        public bool SupportsNeon { get; set; }

        /// <summary>
        /// Whether ARM SVE (Scalable Vector Extension) is supported.
        /// </summary>
        public bool SupportsSve { get; set; }

        /// <summary>
        /// Whether Intel AVX is supported (x86).
        /// </summary>
        public bool SupportsAvx { get; set; }

        /// <summary>
        /// Whether Intel AVX2 is supported.
        /// </summary>
        public bool SupportsAvx2 { get; set; }

        /// <summary>
        /// Whether Intel AVX-512 is supported.
        /// </summary>
        public bool SupportsAvx512 { get; set; }

        /// <summary>
        /// Maximum number of threads available for parallel execution.
        /// </summary>
        public int MaxThreads { get; set; }

        /// <summary>
        /// Cache line size in bytes.
        /// </summary>
        public long CacheLineSize { get; set; }
    }
}
