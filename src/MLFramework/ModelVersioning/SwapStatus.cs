namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Represents the current status of a model version swap operation.
    /// </summary>
    public class SwapStatus
    {
        /// <summary>
        /// Gets or sets the identifier of the model being swapped.
        /// </summary>
        public string ModelId { get; set; }

        /// <summary>
        /// Gets or sets the current version of the model.
        /// </summary>
        public string CurrentVersion { get; set; }

        /// <summary>
        /// Gets or sets the target version to swap to.
        /// </summary>
        public string TargetVersion { get; set; }

        /// <summary>
        /// Gets or sets the current state of the swap operation.
        /// </summary>
        public SwapState State { get; set; }

        /// <summary>
        /// Gets or sets the start time of the swap operation.
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// Gets or sets the number of pending requests during the swap.
        /// </summary>
        public int PendingRequests { get; set; }
    }
}
