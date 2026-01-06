using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Linq;

namespace MLFramework.Data.Metrics
{
    /// <summary>
    /// Central metrics collector for performance monitoring of data loading operations.
    /// Thread-safe and supports enabling/disabling to avoid overhead when not needed.
    /// </summary>
    public class DataLoadingMetrics
    {
        private readonly ConcurrentQueue<TimingRecord> _timingRecords;
        private readonly ConcurrentDictionary<string, PerformanceCounter> _counters;
        private readonly Stopwatch _epochTimer;
        private volatile bool _enabled;

        /// <summary>
        /// Gets a value indicating whether metrics collection is enabled.
        /// </summary>
        public bool Enabled => _enabled;

        /// <summary>
        /// Initializes a new instance of the DataLoadingMetrics class.
        /// </summary>
        /// <param name="enabled">Whether metrics collection is enabled.</param>
        public DataLoadingMetrics(bool enabled = true)
        {
            _timingRecords = new ConcurrentQueue<TimingRecord>();
            _counters = new ConcurrentDictionary<string, PerformanceCounter>();
            _epochTimer = new Stopwatch();
            _enabled = enabled;
        }

        /// <summary>
        /// Records a timing metric.
        /// </summary>
        /// <param name="metricName">The name of the metric.</param>
        /// <param name="duration">The duration to record.</param>
        /// <param name="workerId">The worker ID that performed the operation.</param>
        public void RecordTiming(string metricName, TimeSpan duration, int workerId = -1)
        {
            if (!_enabled)
                return;

            var record = new TimingRecord
            {
                MetricName = metricName,
                Duration = duration,
                Timestamp = DateTime.UtcNow,
                WorkerId = workerId
            };

            _timingRecords.Enqueue(record);

            // Update counter
            var counter = _counters.GetOrAdd(metricName, _ => new PerformanceCounter());
            counter.Record(duration.TotalMilliseconds);
        }

        /// <summary>
        /// Records a counter metric.
        /// </summary>
        /// <param name="metricName">The name of the metric.</param>
        /// <param name="value">The value to record.</param>
        public void RecordCounter(string metricName, double value)
        {
            if (!_enabled)
                return;

            var counter = _counters.GetOrAdd(metricName, _ => new PerformanceCounter());
            counter.Record(value);
        }

        /// <summary>
        /// Gets a summary of all recorded metrics.
        /// </summary>
        /// <returns>A dictionary of metric summaries.</returns>
        public System.Collections.Generic.Dictionary<string, MetricSummary> GetMetricsSummary()
        {
            var summary = new System.Collections.Generic.Dictionary<string, MetricSummary>();

            foreach (var kvp in _counters)
            {
                summary[kvp.Key] = new MetricSummary
                {
                    Count = kvp.Value.Count,
                    Average = kvp.Value.Average,
                    Min = kvp.Value.Min,
                    Max = kvp.Value.Max
                };
            }

            return summary;
        }

        /// <summary>
        /// Starts a new epoch timer.
        /// </summary>
        public void StartEpoch()
        {
            if (!_enabled)
                return;

            _epochTimer.Restart();
        }

        /// <summary>
        /// Ends the current epoch and records the epoch time.
        /// </summary>
        public void EndEpoch()
        {
            if (!_enabled)
                return;

            _epochTimer.Stop();

            var counter = _counters.GetOrAdd("EpochTime", _ => new PerformanceCounter());
            counter.Record(_epochTimer.Elapsed.TotalMilliseconds);
        }

        /// <summary>
        /// Resets all metrics and counters.
        /// </summary>
        public void Reset()
        {
            _timingRecords.Clear();
            _counters.Clear();
            _epochTimer.Reset();
        }

        /// <summary>
        /// Enables or disables metrics collection.
        /// </summary>
        /// <param name="enabled">Whether to enable metrics collection.</param>
        public void SetEnabled(bool enabled)
        {
            _enabled = enabled;
        }
    }
}
