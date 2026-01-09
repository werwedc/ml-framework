using MLFramework.MobileRuntime.Backends.Cpu.Models;

namespace MLFramework.MobileRuntime.Backends.Cpu.Models
{
    /// <summary>
    /// Information about the CPU.
    /// </summary>
    public class CpuInfo
    {
        /// <summary>
        /// CPU vendor (e.g., "AuthenticAMD", "GenuineIntel", "ARM").
        /// </summary>
        public string Vendor { get; set; } = string.Empty;

        /// <summary>
        /// CPU model name.
        /// </summary>
        public string Model { get; set; } = string.Empty;

        /// <summary>
        /// Number of physical cores.
        /// </summary>
        public int CoreCount { get; set; }

        /// <summary>
        /// Number of logical threads.
        /// </summary>
        public int ThreadCount { get; set; }

        /// <summary>
        /// CPU frequency in MHz.
        /// </summary>
        public int FrequencyMHz { get; set; }

        /// <summary>
        /// SIMD and other capabilities of the CPU.
        /// </summary>
        public BackendCapabilities Capabilities { get; set; } = new BackendCapabilities();
    }
}
