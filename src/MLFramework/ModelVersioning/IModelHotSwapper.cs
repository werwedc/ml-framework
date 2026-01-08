using MLFramework.Serving.Routing;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Interface for performing zero-downtime model version swaps with graceful draining and instant rollbacks.
    /// </summary>
    public interface IModelHotSwapper
    {
        /// <summary>
        /// Swaps from one model version to another with zero downtime.
        /// </summary>
        /// <param name="modelId">The identifier of the model.</param>
        /// <param name="fromVersion">The source version to swap from.</param>
        /// <param name="toVersion">The target version to swap to.</param>
        /// <returns>A <see cref="Task"/> representing the swap operation result.</returns>
        Task<SwapResult> SwapVersion(string modelId, string fromVersion, string toVersion);

        /// <summary>
        /// Rolls back to a previous version immediately.
        /// </summary>
        /// <param name="modelId">The identifier of the model.</param>
        /// <param name="targetVersion">The target version to roll back to.</param>
        /// <returns>A <see cref="Task"/> representing the rollback operation result.</returns>
        Task<RollbackResult> RollbackVersion(string modelId, string targetVersion);

        /// <summary>
        /// Performs a health check on a specific model version.
        /// </summary>
        /// <param name="modelId">The identifier of the model.</param>
        /// <param name="version">The version to check health for.</param>
        /// <returns>The health check result.</returns>
        HealthCheckResult CheckVersionHealth(string modelId, string version);

        /// <summary>
        /// Drains in-flight requests from a specific version within a timeout.
        /// </summary>
        /// <param name="modelId">The identifier of the model.</param>
        /// <param name="version">The version to drain.</param>
        /// <param name="timeout">The timeout duration for draining.</param>
        /// <returns>True if draining completed successfully within timeout, otherwise false.</returns>
        Task<bool> DrainVersion(string modelId, string version, TimeSpan timeout);

        /// <summary>
        /// Gets the current swap status for a model.
        /// </summary>
        /// <param name="modelId">The identifier of the model.</param>
        /// <returns>The current swap status.</returns>
        SwapStatus GetSwapStatus(string modelId);
    }
}
