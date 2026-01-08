namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Represents the result of a model version rollback operation.
    /// </summary>
    public class RollbackResult
    {
        /// <summary>
        /// Gets or sets a value indicating whether the rollback operation was successful.
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Gets or sets a message describing the result of the rollback operation.
        /// </summary>
        public string Message { get; set; }

        /// <summary>
        /// Gets or sets the time when the rollback was performed.
        /// </summary>
        public DateTime RollbackTime { get; set; }

        /// <summary>
        /// Gets or sets the previous version that was active before the rollback.
        /// </summary>
        public string PreviousVersion { get; set; }

        /// <summary>
        /// Gets or sets the new version after the rollback.
        /// </summary>
        public string NewVersion { get; set; }
    }
}
