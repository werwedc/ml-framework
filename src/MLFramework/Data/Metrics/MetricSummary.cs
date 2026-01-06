namespace MLFramework.Data.Metrics
{
    /// <summary>
    /// Summary statistics for a metric.
    /// </summary>
    public class MetricSummary
    {
        /// <summary>
        /// Gets or sets the number of recorded values.
        /// </summary>
        public long Count { get; set; }

        /// <summary>
        /// Gets or sets the average of all recorded values.
        /// </summary>
        public double Average { get; set; }

        /// <summary>
        /// Gets or sets the minimum recorded value.
        /// </summary>
        public double Min { get; set; }

        /// <summary>
        /// Gets or sets the maximum recorded value.
        /// </summary>
        public double Max { get; set; }

        /// <summary>
        /// Gets the total sum of all recorded values.
        /// </summary>
        public double Total => Average * Count;
    }
}
