namespace MLFramework.MobileRuntime
{
    /// <summary>
    /// Hardware backend types supported by the mobile runtime.
    /// </summary>
    public enum BackendType
    {
        /// <summary>
        /// CPU-based execution.
        /// </summary>
        CPU,

        /// <summary>
        /// GPU-based execution (e.g., Vulkan, Metal, OpenCL).
        /// </summary>
        GPU,

        /// <summary>
        /// Neural Processing Unit (NPU) execution.
        /// </summary>
        NPU,

        /// <summary>
        /// Automatically select the best available backend.
        /// </summary>
        Auto
    }
}
