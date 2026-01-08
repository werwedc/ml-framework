namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Represents the result of a model version swap operation.
    /// </summary>
    public class SwapResult
    {
        /// <summary>
        /// Gets or sets a value indicating whether the swap operation was successful.
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Gets or sets a message describing the result of the swap operation.
        /// </summary>
        public string Message { get; set; }

        /// <summary>
        /// Gets or sets the start time of the swap operation.
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// Gets or sets the end time of the swap operation.
        /// </summary>
        public DateTime EndTime { get; set; }

        /// <summary>
        /// Gets or sets the number of requests drained during the swap.
        /// </summary>
        public int RequestsDrained { get; set; }

        /// <summary>
        /// Gets or sets the number of requests remaining when the swap completed.
        /// </summary>
        public int RequestsRemaining { get; set; }
    }
}
