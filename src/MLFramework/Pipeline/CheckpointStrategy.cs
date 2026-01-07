namespace MLFramework.Pipeline
{
    /// <summary>
    /// Strategy for activation checkpointing
    /// </summary>
    public enum CheckpointStrategy
    {
        /// <summary>
        /// Store all activations (maximum memory, fastest backward)
        /// </summary>
        StoreAll,

        /// <summary>
        /// Recompute all activations during backward (minimum memory, slowest)
        /// </summary>
        RecomputeAll,

        /// <summary>
        /// Store every Nth activation (balanced)
        /// </summary>
        Selective,

        /// <summary>
        /// Store activations based on memory threshold
        /// </summary>
        MemoryBased
    }
}
