namespace MLFramework.Pipeline
{
    /// <summary>
    /// Statistics about pipeline execution
    /// </summary>
    public class PipelineStats
    {
        /// <summary>
        /// Total time for forward pass (ms)
        /// </summary>
        public float ForwardTime { get; }

        /// <summary>
        /// Total time for backward pass (ms)
        /// </summary>
        public float BackwardTime { get; }

        /// <summary>
        /// Bubble time (idle time) during forward pass (ms)
        /// </summary>
        public float ForwardBubbleTime { get; }

        /// <summary>
        /// Bubble time (idle time) during backward pass (ms)
        /// </summary>
        public float BackwardBubbleTime { get; }

        /// <summary>
        /// Device utilization (0.0 to 1.0)
        /// </summary>
        public float Utilization =>
            (ForwardTime + BackwardTime) > 0
                ? 1.0f - (ForwardBubbleTime + BackwardBubbleTime) / (ForwardTime + BackwardTime)
                : 0.0f;

        public PipelineStats(float forwardTime, float backwardTime, float forwardBubbleTime, float backwardBubbleTime)
        {
            ForwardTime = forwardTime;
            BackwardTime = backwardTime;
            ForwardBubbleTime = forwardBubbleTime;
            BackwardBubbleTime = backwardBubbleTime;
        }
    }
}
