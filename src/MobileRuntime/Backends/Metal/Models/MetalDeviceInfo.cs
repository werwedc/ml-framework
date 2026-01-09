namespace MobileRuntime.Backends.Metal
{
    /// <summary>
    /// Information about the Metal device
    /// </summary>
    public class MetalDeviceInfo
    {
        /// <summary>
        /// Gets or sets the device name
        /// </summary>
        public string DeviceName { get; set; }

        /// <summary>
        /// Gets or sets the device family ID
        /// </summary>
        public uint FamilyId { get; set; }

        /// <summary>
        /// Gets or sets the recommended maximum working set size
        /// </summary>
        public uint RecommendedMaxWorkingSetSize { get; set; }

        /// <summary>
        /// Gets or sets whether the device has unified memory
        /// </summary>
        public bool HasUnifiedMemory { get; set; }

        /// <summary>
        /// Gets or sets the backend capabilities
        /// </summary>
        public MetalBackendCapabilities Capabilities { get; set; }
    }
}
