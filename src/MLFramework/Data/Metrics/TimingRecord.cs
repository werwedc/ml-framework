using System;

namespace MLFramework.Data.Metrics
{
    /// <summary>
    /// Represents a single timing record for performance tracking.
    /// </summary>
    public class TimingRecord
    {
        /// <summary>
        /// Gets or sets the name of the metric being recorded.
        /// </summary>
        public string MetricName { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the duration of the operation.
        /// </summary>
        public TimeSpan Duration { get; set; }

        /// <summary>
        /// Gets or sets the timestamp when the record was created.
        /// </summary>
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// Gets or sets the worker ID that performed the operation.
        /// </summary>
        public int WorkerId { get; set; }
    }
}
