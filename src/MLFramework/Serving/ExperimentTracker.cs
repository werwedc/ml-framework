using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Serving
{
    /// <summary>
    /// Thread-safe implementation of experiment tracking for A/B testing.
    /// Tracks performance metrics per model version across experiments.
    /// </summary>
    public class ExperimentTracker : IExperimentTracker
    {
        private readonly ConcurrentDictionary<string, ExperimentState> _experiments;
        private readonly object _experimentLock = new object();

        /// <summary>
        /// Internal state for tracking an experiment.
        /// </summary>
        private class ExperimentState
        {
            public string ExperimentId { get; set; }
            public string ModelName { get; set; }
            public DateTime StartTime { get; set; }
            public DateTime? EndTime { get; set; }
            public ConcurrentDictionary<string, VersionMetrics> VersionMetrics { get; set; }
            public bool IsActive { get; set; }
        }

        /// <summary>
        /// Internal metrics tracking for a specific version.
        /// </summary>
        private class VersionMetrics
        {
            public int RequestCount { get; set; }
            public int SuccessCount { get; set; }
            public int ErrorCount { get; set; }
            public double TotalLatencyMs { get; set; }
            public ConcurrentBag<double> LatencySamples { get; set; }
            public ConcurrentDictionary<string, List<double>> CustomMetrics { get; set; }

            public VersionMetrics()
            {
                LatencySamples = new ConcurrentBag<double>();
                CustomMetrics = new ConcurrentDictionary<string, List<double>>();
            }
        }

        /// <summary>
        /// Initializes a new instance of the ExperimentTracker.
        /// </summary>
        public ExperimentTracker()
        {
            _experiments = new ConcurrentDictionary<string, ExperimentState>();
        }

        /// <inheritdoc/>
        public void StartExperiment(string experimentId, string modelName, Dictionary<string, float> versionTraffic)
        {
            if (string.IsNullOrWhiteSpace(experimentId))
                throw new ArgumentException("Experiment ID cannot be null or empty.", nameof(experimentId));

            if (string.IsNullOrWhiteSpace(modelName))
                throw new ArgumentException("Model name cannot be null or empty.", nameof(modelName));

            if (versionTraffic == null || versionTraffic.Count == 0)
                throw new ArgumentException("Version traffic dictionary cannot be null or empty.", nameof(versionTraffic));

            // Validate traffic splits sum to approximately 1.0 (with floating point tolerance)
            var totalTraffic = versionTraffic.Values.Sum();
            if (Math.Abs(totalTraffic - 1.0f) > 0.01f)
                throw new ArgumentException($"Version traffic splits must sum to 1.0 (got {totalTraffic}).", nameof(versionTraffic));

            lock (_experimentLock)
            {
                if (_experiments.ContainsKey(experimentId))
                    throw new InvalidOperationException($"Experiment '{experimentId}' already exists.");

                var experiment = new ExperimentState
                {
                    ExperimentId = experimentId,
                    ModelName = modelName,
                    StartTime = DateTime.UtcNow,
                    IsActive = true,
                    VersionMetrics = new ConcurrentDictionary<string, VersionMetrics>()
                };

                // Initialize metrics for each version
                foreach (var version in versionTraffic.Keys)
                {
                    experiment.VersionMetrics[version] = new VersionMetrics();
                }

                if (!_experiments.TryAdd(experimentId, experiment))
                {
                    throw new InvalidOperationException($"Failed to add experiment '{experimentId}'.");
                }
            }
        }

        /// <inheritdoc/>
        public void EndExperiment(string experimentId)
        {
            if (string.IsNullOrWhiteSpace(experimentId))
                throw new ArgumentException("Experiment ID cannot be null or empty.", nameof(experimentId));

            lock (_experimentLock)
            {
                if (!_experiments.TryGetValue(experimentId, out var experiment))
                    throw new KeyNotFoundException($"Experiment '{experimentId}' not found.");

                if (!experiment.IsActive)
                    throw new InvalidOperationException($"Experiment '{experimentId}' is already ended.");

                experiment.IsActive = false;
                experiment.EndTime = DateTime.UtcNow;
            }
        }

        /// <inheritdoc/>
        public void RecordInference(string experimentId, string version, double latencyMs, bool success, Dictionary<string, double> customMetrics = null)
        {
            if (string.IsNullOrWhiteSpace(experimentId))
                throw new ArgumentException("Experiment ID cannot be null or empty.", nameof(experimentId));

            if (string.IsNullOrWhiteSpace(version))
                throw new ArgumentException("Version cannot be null or empty.", nameof(version));

            if (latencyMs < 0)
                throw new ArgumentException("Latency must be non-negative.", nameof(latencyMs));

            if (!_experiments.TryGetValue(experimentId, out var experiment))
                throw new KeyNotFoundException($"Experiment '{experimentId}' not found.");

            if (!experiment.IsActive)
                throw new InvalidOperationException($"Experiment '{experimentId}' has ended.");

            if (!experiment.VersionMetrics.TryGetValue(version, out var versionMetrics))
                throw new KeyNotFoundException($"Version '{version}' not found in experiment '{experimentId}'.");

            // Record basic metrics
            versionMetrics.RequestCount++;
            versionMetrics.TotalLatencyMs += latencyMs;

            if (success)
            {
                versionMetrics.SuccessCount++;
            }
            else
            {
                versionMetrics.ErrorCount++;
            }

            // Record latency sample for percentile calculation
            versionMetrics.LatencySamples.Add(latencyMs);

            // Record custom metrics if provided
            if (customMetrics != null)
            {
                foreach (var kvp in customMetrics)
                {
                    var metricsList = versionMetrics.CustomMetrics.GetOrAdd(kvp.Key, new List<double>());
                    lock (metricsList)
                    {
                        metricsList.Add(kvp.Value);
                    }
                }
            }
        }

        /// <inheritdoc/>
        public ExperimentMetrics GetMetrics(string experimentId, string version)
        {
            if (string.IsNullOrWhiteSpace(experimentId))
                throw new ArgumentException("Experiment ID cannot be null or empty.", nameof(experimentId));

            if (string.IsNullOrWhiteSpace(version))
                throw new ArgumentException("Version cannot be null or empty.", nameof(version));

            if (!_experiments.TryGetValue(experimentId, out var experiment))
                throw new KeyNotFoundException($"Experiment '{experimentId}' not found.");

            if (!experiment.VersionMetrics.TryGetValue(version, out var versionMetrics))
                throw new KeyNotFoundException($"Version '{version}' not found in experiment '{experimentId}'.");

            return BuildExperimentMetrics(experiment, version, versionMetrics);
        }

        /// <inheritdoc/>
        public Dictionary<string, ExperimentMetrics> GetAllMetrics(string experimentId)
        {
            if (string.IsNullOrWhiteSpace(experimentId))
                throw new ArgumentException("Experiment ID cannot be null or empty.", nameof(experimentId));

            if (!_experiments.TryGetValue(experimentId, out var experiment))
                throw new KeyNotFoundException($"Experiment '{experimentId}' not found.");

            var result = new Dictionary<string, ExperimentMetrics>();
            foreach (var kvp in experiment.VersionMetrics)
            {
                result[kvp.Key] = BuildExperimentMetrics(experiment, kvp.Key, kvp.Value);
            }

            return result;
        }

        /// <inheritdoc/>
        public Dictionary<string, double> CompareVersions(string experimentId)
        {
            if (string.IsNullOrWhiteSpace(experimentId))
                throw new ArgumentException("Experiment ID cannot be null or empty.", nameof(experimentId));

            if (!_experiments.TryGetValue(experimentId, out var experiment))
                throw new KeyNotFoundException($"Experiment '{experimentId}' not found.");

            var allMetrics = GetAllMetrics(experimentId);
            var comparisons = new Dictionary<string, double>();

            // Compare each version against every other version
            var versions = allMetrics.Keys.ToList();
            for (int i = 0; i < versions.Count; i++)
            {
                for (int j = i + 1; j < versions.Count; j++)
                {
                    var v1 = versions[i];
                    var v2 = versions[j];
                    var m1 = allMetrics[v1];
                    var m2 = allMetrics[v2];

                    // Calculate latency difference (v2 - v1)
                    var latencyDiff = m2.AverageLatencyMs - m1.AverageLatencyMs;
                    comparisons[$"{v1}_vs_{v2}_latency_diff_ms"] = latencyDiff;

                    // Calculate success rate difference (v2 - v1)
                    var successRateDiff = m2.SuccessRate - m1.SuccessRate;
                    comparisons[$"{v1}_vs_{v2}_success_rate_diff_pct"] = successRateDiff;

                    // Calculate error rate difference (v2 - v1)
                    var errorRateDiff = m2.ErrorRate - m1.ErrorRate;
                    comparisons[$"{v1}_vs_{v2}_error_rate_diff_pct"] = errorRateDiff;

                    // Calculate t-test statistic for latency comparison
                    if (m1.RequestCount > 1 && m2.RequestCount > 1)
                    {
                        var tStat = CalculateTStatistic(m1, m2);
                        comparisons[$"{v1}_vs_{v2}_t_statistic"] = tStat;
                    }
                }
            }

            return comparisons;
        }

        /// <summary>
        /// Builds an ExperimentMetrics object from internal tracking state.
        /// </summary>
        private ExperimentMetrics BuildExperimentMetrics(ExperimentState experiment, string version, VersionMetrics versionMetrics)
        {
            var metrics = new ExperimentMetrics
            {
                ExperimentId = experiment.ExperimentId,
                ModelName = experiment.ModelName,
                Version = version,
                StartTime = experiment.StartTime,
                EndTime = experiment.EndTime,
                RequestCount = versionMetrics.RequestCount,
                SuccessCount = versionMetrics.SuccessCount,
                ErrorCount = versionMetrics.ErrorCount,
                AverageLatencyMs = versionMetrics.RequestCount > 0 ? versionMetrics.TotalLatencyMs / versionMetrics.RequestCount : 0
            };

            // Calculate percentiles from latency samples
            var samples = versionMetrics.LatencySamples.ToList();
            if (samples.Count > 0)
            {
                samples.Sort();
                metrics.P50LatencyMs = CalculatePercentile(samples, 50);
                metrics.P95LatencyMs = CalculatePercentile(samples, 95);
                metrics.P99LatencyMs = CalculatePercentile(samples, 99);
            }

            // Aggregate custom metrics
            foreach (var kvp in versionMetrics.CustomMetrics)
            {
                var values = kvp.Value;
                double avgValue = values.Count > 0 ? values.Average() : 0;
                metrics.CustomMetrics[kvp.Key] = avgValue;
            }

            return metrics;
        }

        /// <summary>
        /// Calculates a percentile from sorted samples.
        /// </summary>
        private double CalculatePercentile(List<double> sortedSamples, double percentile)
        {
            if (sortedSamples.Count == 0)
                return 0;

            int index = (int)Math.Ceiling(percentile / 100.0 * sortedSamples.Count) - 1;
            index = Math.Max(0, Math.Min(index, sortedSamples.Count - 1));
            return sortedSamples[index];
        }

        /// <summary>
        /// Calculates the t-statistic for comparing two sets of metrics.
        /// </summary>
        private double CalculateTStatistic(ExperimentMetrics m1, ExperimentMetrics m2)
        {
            // Simplified t-test assuming equal variances
            var n1 = m1.RequestCount;
            var n2 = m2.RequestCount;
            var mean1 = m1.AverageLatencyMs;
            var mean2 = m2.AverageLatencyMs;

            // Calculate pooled variance (simplified)
            var var1 = Math.Pow(m1.P95LatencyMs - m1.P50LatencyMs, 2); // Use range as variance proxy
            var var2 = Math.Pow(m2.P95LatencyMs - m2.P50LatencyMs, 2);

            var pooledVariance = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
            var standardError = Math.Sqrt(pooledVariance * (1.0 / n1 + 1.0 / n2));

            if (standardError == 0)
                return 0;

            return (mean1 - mean2) / standardError;
        }
    }
}
