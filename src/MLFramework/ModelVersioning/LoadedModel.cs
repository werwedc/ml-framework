namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Internal class representing a loaded model instance with tracking information.
    /// </summary>
    internal class LoadedModel
    {
        /// <summary>
        /// Gets or sets the identifier of the model.
        /// </summary>
        public string ModelId { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the version of the model.
        /// </summary>
        public string Version { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the model instance (generic wrapper for any model type).
        /// </summary>
        public object? ModelInstance { get; set; }

        /// <summary>
        /// Gets or sets the time when the model was loaded.
        /// </summary>
        public DateTime LoadTime { get; set; }

        /// <summary>
        /// Gets or sets the estimated memory usage in bytes.
        /// </summary>
        public long MemoryUsageBytes { get; set; }

        /// <summary>
        /// Gets the number of requests processed by this version.
        /// Note: For thread-safe updates, use Interlocked.Increment.
        /// </summary>
        public int RequestCount;

        /// <summary>
        /// Gets or sets a value indicating whether the model is currently warming up.
        /// </summary>
        public bool IsWarmingUp { get; set; }
    }
}
