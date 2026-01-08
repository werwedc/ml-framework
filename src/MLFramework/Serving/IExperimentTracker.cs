using System;
using System.Collections.Generic;

namespace MLFramework.Serving
{
    /// <summary>
    /// Interface for tracking A/B testing experiments and collecting performance metrics per model version.
    /// </summary>
    public interface IExperimentTracker
    {
        /// <summary>
        /// Starts a new experiment with the specified traffic distribution across versions.
        /// </summary>
        /// <param name="experimentId">Unique identifier for the experiment.</param>
        /// <param name="modelName">Name of the model being tested.</param>
        /// <param name="versionTraffic">Dictionary mapping version IDs to traffic split percentages.</param>
        void StartExperiment(string experimentId, string modelName, Dictionary<string, float> versionTraffic);

        /// <summary>
        /// Ends an active experiment and finalizes all metrics.
        /// </summary>
        /// <param name="experimentId">Experiment ID to end.</param>
        void EndExperiment(string experimentId);

        /// <summary>
        /// Records an inference result for a specific version within an experiment.
        /// </summary>
        /// <param name="experimentId">Experiment ID.</param>
        /// <param name="version">Model version.</param>
        /// <param name="latencyMs">Latency in milliseconds.</param>
        /// <param name="success">Whether the inference was successful.</param>
        /// <param name="customMetrics">Optional custom metrics to record.</param>
        void RecordInference(string experimentId, string version, double latencyMs, bool success, Dictionary<string, double> customMetrics = null);

        /// <summary>
        /// Gets aggregated metrics for a specific version within an experiment.
        /// </summary>
        /// <param name="experimentId">Experiment ID.</param>
        /// <param name="version">Model version.</param>
        /// <returns>Aggregated metrics for the specified version.</returns>
        ExperimentMetrics GetMetrics(string experimentId, string version);

        /// <summary>
        /// Gets all metrics for all versions within an experiment.
        /// </summary>
        /// <param name="experimentId">Experiment ID.</param>
        /// <returns>Dictionary mapping version IDs to their metrics.</returns>
        Dictionary<string, ExperimentMetrics> GetAllMetrics(string experimentId);

        /// <summary>
        /// Compares performance between versions in an experiment.
        /// </summary>
        /// <param name="experimentId">Experiment ID.</param>
        /// <returns>Dictionary comparing versions (e.g., version pairs to performance differences).</returns>
        Dictionary<string, double> CompareVersions(string experimentId);
    }
}
