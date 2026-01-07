namespace MLFramework.Pipeline
{
    /// <summary>
    /// Strategy for gradient synchronization across pipeline stages
    /// </summary>
    public enum GradientSyncMode
    {
        /// <summary>
        /// No synchronization (each stage has its own copy)
        /// </summary>
        None,

        /// <summary>
        /// Average gradients across all stages (each stage has same model)
        /// </summary>
        Average,

        /// <summary>
        /// Sum gradients across all stages
        /// </summary>
        Sum,

        /// <summary>
        /// Each stage updates only its parameters (model partitioning)
        /// </summary>
        StageWise
    }
}
