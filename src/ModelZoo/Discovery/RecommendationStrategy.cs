namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Different strategies for model recommendation based on use case priorities.
    /// </summary>
    public enum RecommendationStrategy
    {
        /// <summary>
        /// Prioritize highest accuracy models.
        /// </summary>
        AccuracyFirst,

        /// <summary>
        /// Prioritize lowest latency models.
        /// </summary>
        PerformanceFirst,

        /// <summary>
        /// Balance accuracy and performance.
        /// </summary>
        Balanced,

        /// <summary>
        /// Prioritize smallest models for memory-constrained environments.
        /// </summary>
        MemoryConstrained,

        /// <summary>
        /// Prioritize models suitable for edge devices.
        /// </summary>
        EdgeDeployment
    }
}
