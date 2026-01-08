namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Represents the difference between two metric values.
    /// </summary>
    public class MetricDelta
    {
        /// <summary>
        /// Absolute difference between the two values.
        /// </summary>
        public double AbsoluteDifference { get; set; }

        /// <summary>
        /// Percentage change from baseline to new value.
        /// </summary>
        public double PercentageChange { get; set; }

        /// <summary>
        /// Direction of change: "better", "worse", or "neutral".
        /// </summary>
        public string Direction { get; set; }

        /// <summary>
        /// Creates a metric delta for latency comparison (lower is better).
        /// </summary>
        public static MetricDelta CreateForLatency(double baseline, double newValue)
        {
            double absDiff = newValue - baseline;
            double pctChange = baseline != 0 ? (absDiff / baseline) * 100 : 0;

            string direction;
            if (Math.Abs(absDiff) < 0.01) // Use epsilon for floating point comparison
            {
                direction = "neutral";
            }
            else if (absDiff < 0)
            {
                direction = "better"; // Lower latency is better
            }
            else
            {
                direction = "worse"; // Higher latency is worse
            }

            return new MetricDelta
            {
                AbsoluteDifference = absDiff,
                PercentageChange = pctChange,
                Direction = direction
            };
        }

        /// <summary>
        /// Creates a metric delta for error rate comparison (lower is better).
        /// </summary>
        public static MetricDelta CreateForErrorRate(double baseline, double newValue)
        {
            double absDiff = newValue - baseline;
            double pctChange = baseline != 0 ? (absDiff / baseline) * 100 : 0;

            string direction;
            if (Math.Abs(absDiff) < 0.0001) // Use epsilon for error rate
            {
                direction = "neutral";
            }
            else if (absDiff < 0)
            {
                direction = "better"; // Lower error rate is better
            }
            else
            {
                direction = "worse"; // Higher error rate is worse
            }

            return new MetricDelta
            {
                AbsoluteDifference = absDiff,
                PercentageChange = pctChange,
                Direction = direction
            };
        }

        /// <summary>
        /// Creates a metric delta for throughput comparison (higher is better).
        /// </summary>
        public static MetricDelta CreateForThroughput(double baseline, double newValue)
        {
            double absDiff = newValue - baseline;
            double pctChange = baseline != 0 ? (absDiff / baseline) * 100 : 0;

            string direction;
            if (Math.Abs(absDiff) < 0.01) // Use epsilon for throughput
            {
                direction = "neutral";
            }
            else if (absDiff > 0)
            {
                direction = "better"; // Higher throughput is better
            }
            else
            {
                direction = "worse"; // Lower throughput is worse
            }

            return new MetricDelta
            {
                AbsoluteDifference = absDiff,
                PercentageChange = pctChange,
                Direction = direction
            };
        }

        /// <summary>
        /// Creates a copy of this metric delta.
        /// </summary>
        public MetricDelta Clone()
        {
            return new MetricDelta
            {
                AbsoluteDifference = AbsoluteDifference,
                PercentageChange = PercentageChange,
                Direction = Direction
            };
        }
    }
}
