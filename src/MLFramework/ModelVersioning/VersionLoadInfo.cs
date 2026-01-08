namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Contains information about the load status of a model version.
    /// </summary>
    public class VersionLoadInfo
    {
        /// <summary>
        /// Gets or sets the identifier of the model.
        /// </summary>
        public string ModelId { get; set; }

        /// <summary>
        /// Gets or sets the version of the model.
        /// </summary>
        public string Version { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether the version is loaded.
        /// </summary>
        public bool IsLoaded { get; set; }

        /// <summary>
        /// Gets or sets the time when the version was loaded.
        /// </summary>
        public DateTime LoadTime { get; set; }

        /// <summary>
        /// Gets or sets the memory usage of the version in bytes.
        /// </summary>
        public long MemoryUsageBytes { get; set; }

        /// <summary>
        /// Gets or sets the number of requests handled by this version.
        /// </summary>
        public int RequestCount { get; set; }

        /// <summary>
        /// Gets or sets the status of the version.
        /// </summary>
        public string Status { get; set; }
    }
}
