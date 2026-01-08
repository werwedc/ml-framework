namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Represents the lifecycle state of a model version.
    /// </summary>
    public enum LifecycleState
    {
        /// <summary>
        /// Model is in draft state, not ready for use.
        /// </summary>
        Draft,

        /// <summary>
        /// Model is staged for testing and validation.
        /// </summary>
        Staging,

        /// <summary>
        /// Model is in production and serving requests.
        /// </summary>
        Production,

        /// <summary>
        /// Model has been archived and is no longer in use.
        /// </summary>
        Archived
    }
}
