namespace MobileRuntime
{
    /// <summary>
    /// Hardware backend types supported by the mobile runtime.
    /// </summary>
    public enum BackendType
    {
        /// <summary>
        /// CPU-based execution with ARM NEON/SVE optimization.
        /// </summary>
        Cpu,

        /// <summary>
        /// Metal-based execution for iOS.
        /// </summary>
        Metal,

        /// <summary>
        /// Vulkan-based execution for Android.
        /// </summary>
        Vulkan
    }
}
