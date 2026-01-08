namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Represents the current state of a model version swap operation.
    /// </summary>
    public enum SwapState
    {
        /// <summary>
        /// No swap operation is in progress.
        /// </summary>
        Idle,

        /// <summary>
        /// The system is draining in-flight requests from the source version.
        /// </summary>
        Draining,

        /// <summary>
        /// The system is actively swapping between versions.
        /// </summary>
        Swapping,

        /// <summary>
        /// The swap operation has completed successfully.
        /// </summary>
        Completed,

        /// <summary>
        /// The swap operation has failed.
        /// </summary>
        Failed
    }
}
