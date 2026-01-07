namespace MLFramework.Pipeline
{
    /// <summary>
    /// Configuration for pipeline optimizer
    /// </summary>
    public class PipelineOptimizerConfig
    {
        /// <summary>
        /// Gradient synchronization mode
        /// </summary>
        public GradientSyncMode SyncMode { get; set; } = GradientSyncMode.Average;

        /// <summary>
        /// Whether to synchronize gradients before optimizer step
        /// </summary>
        public bool SynchronizeGradients { get; set; } = true;

        /// <summary>
        /// Whether to broadcast parameters after optimizer step
        /// </summary>
        public bool BroadcastParameters { get; set; } = true;

        /// <summary>
        /// Communication timeout in milliseconds
        /// </summary>
        public int CommunicationTimeoutMs { get; set; } = 30000;

        /// <summary>
        /// Validate the configuration
        /// </summary>
        public void Validate()
        {
            if (CommunicationTimeoutMs <= 0)
            {
                throw new System.InvalidOperationException("Communication timeout must be positive");
            }
        }
    }
}
