namespace MLFramework.MobileRuntime
{
    /// <summary>
    /// Information about the mobile runtime.
    /// </summary>
    public class RuntimeInfo
    {
        /// <summary>
        /// Runtime version.
        /// </summary>
        public string Version { get; set; } = string.Empty;

        /// <summary>
        /// Supported hardware backends.
        /// </summary>
        public BackendType SupportedBackends { get; set; }

        /// <summary>
        /// Platform information (e.g., "iOS", "Android", "Windows").
        /// </summary>
        public string Platform { get; set; } = string.Empty;

        /// <summary>
        /// Device-specific information.
        /// </summary>
        public string DeviceInfo { get; set; } = string.Empty;
    }
}
