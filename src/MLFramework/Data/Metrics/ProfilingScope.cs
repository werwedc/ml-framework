using System;
using System.Diagnostics;

namespace MLFramework.Data.Metrics
{
    /// <summary>
    /// A disposable scope for automatic timing recording.
    /// </summary>
    public class ProfilingScope : IDisposable
    {
        private readonly DataLoadingMetrics _metrics;
        private readonly string _metricName;
        private readonly int _workerId;
        private readonly Stopwatch _stopwatch;
        private bool _disposed;

        /// <summary>
        /// Initializes a new instance of the ProfilingScope class.
        /// </summary>
        /// <param name="metrics">The metrics collector to use.</param>
        /// <param name="metricName">The name of the metric to record.</param>
        /// <param name="workerId">The worker ID performing the operation.</param>
        public ProfilingScope(DataLoadingMetrics metrics, string metricName, int workerId = -1)
        {
            _metrics = metrics;
            _metricName = metricName;
            _workerId = workerId;
            _stopwatch = Stopwatch.StartNew();
        }

        /// <summary>
        /// Disposes the scope and records the elapsed time.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;

            _disposed = true;
            _stopwatch.Stop();
            _metrics.RecordTiming(_metricName, _stopwatch.Elapsed, _workerId);
        }
    }

    /// <summary>
    /// Extension methods for DataLoadingMetrics.
    /// </summary>
    public static class DataLoadingMetricsExtensions
    {
        /// <summary>
        /// Creates a profiling scope for automatic timing.
        /// </summary>
        /// <param name="metrics">The metrics collector.</param>
        /// <param name="metricName">The name of the metric.</param>
        /// <param name="workerId">The worker ID performing the operation.</param>
        /// <returns>A profiling scope that records time on disposal.</returns>
        public static IDisposable Profile(this DataLoadingMetrics metrics, string metricName, int workerId = -1)
        {
            return new ProfilingScope(metrics, metricName, workerId);
        }
    }
}
