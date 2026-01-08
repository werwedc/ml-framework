using System;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Interface for monitoring model version metrics, comparing versions, and alerting on anomalies.
    /// </summary>
    public interface IVersionMonitor
    {
        /// <summary>
        /// Records a metric sample for a specific model version.
        /// </summary>
        /// <param name="modelId">The model identifier.</param>
        /// <param name="version">The version tag.</param>
        /// <param name="sample">The metric sample to record.</param>
        void RecordMetric(string modelId, string version, MetricSample sample);

        /// <summary>
        /// Gets aggregated metrics for a specific model version.
        /// </summary>
        /// <param name="modelId">The model identifier.</param>
        /// <param name="version">The version tag.</param>
        /// <returns>Aggregated metrics for the version.</returns>
        VersionMetrics GetMetrics(string modelId, string version);

        /// <summary>
        /// Compares metrics between two versions of a model.
        /// </summary>
        /// <param name="modelId">The model identifier.</param>
        /// <param name="v1">The first version (baseline).</param>
        /// <param name="v2">The second version (comparison).</param>
        /// <returns>A comparison of metrics between the versions.</returns>
        MetricComparison CompareVersions(string modelId, string v1, string v2);

        /// <summary>
        /// Subscribes to version alerts.
        /// </summary>
        /// <param name="callback">Callback to invoke when an alert is triggered.</param>
        void SubscribeToAlerts(Action<VersionAlert> callback);

        /// <summary>
        /// Unsubscribes from version alerts.
        /// </summary>
        /// <param name="callback">The callback to remove.</param>
        void UnsubscribeFromAlerts(Action<VersionAlert> callback);

        /// <summary>
        /// Clears all metrics for a specific model version.
        /// </summary>
        /// <param name="modelId">The model identifier.</param>
        /// <param name="version">The version tag.</param>
        void ClearMetrics(string modelId, string version);
    }
}
