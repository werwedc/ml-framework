using System;
using System.Collections.Generic;

namespace MLFramework.Serving.Traffic
{
    /// <summary>
    /// Interface for managing traffic distribution between multiple model versions.
    /// Supports percentage-based traffic allocation for A/B testing and canary deployments.
    /// </summary>
    public interface ITrafficSplitter
    {
        /// <summary>
        /// Configures the traffic split for a specific model.
        /// </summary>
        /// <param name="modelName">The name of the model</param>
        /// <param name="versionPercentages">Dictionary mapping version names to their percentage allocations (0.0-1.0)</param>
        /// <exception cref="ArgumentException">Thrown when percentages don't sum to 1.0, contain negative values, or are out of range</exception>
        /// <exception cref="ArgumentNullException">Thrown when modelName or versionPercentages is null</exception>
        void SetTrafficSplit(string modelName, Dictionary<string, float> versionPercentages);

        /// <summary>
        /// Selects a model version for the given request based on configured traffic split.
        /// Uses deterministic hashing to ensure the same request ID consistently routes to the same version.
        /// </summary>
        /// <param name="modelName">The name of the model</param>
        /// <param name="requestId">Unique identifier for the request</param>
        /// <returns>The selected version name</returns>
        /// <exception cref="KeyNotFoundException">Thrown when no traffic split is configured for the model</exception>
        string SelectVersion(string modelName, string requestId);

        /// <summary>
        /// Retrieves the current traffic split configuration for a model.
        /// </summary>
        /// <param name="modelName">The name of the model</param>
        /// <returns>The traffic split configuration, or null if not configured</returns>
        TrafficSplitConfig GetTrafficSplit(string modelName);

        /// <summary>
        /// Removes the traffic split configuration for a model.
        /// </summary>
        /// <param name="modelName">The name of the model</param>
        void ClearTrafficSplit(string modelName);

        /// <summary>
        /// Gets the allocation percentage for a specific model version.
        /// </summary>
        /// <param name="modelName">The name of the model</param>
        /// <param name="version">The version name</param>
        /// <returns>The allocation percentage (0.0-1.0), or 0.0 if not found</returns>
        float GetVersionAllocation(string modelName, string version);
    }
}
