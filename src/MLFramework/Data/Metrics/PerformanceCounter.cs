using System;
using System.Threading;

namespace MLFramework.Data.Metrics
{
    /// <summary>
    /// Thread-safe performance counter for tracking statistics.
    /// </summary>
    public class PerformanceCounter
    {
        private long _count;
        private double _sum;
        private double _min = double.MaxValue;
        private double _max = double.MinValue;

        /// <summary>
        /// Records a value to the counter.
        /// </summary>
        /// <param name="value">The value to record.</param>
        public void Record(double value)
        {
            Interlocked.Increment(ref _count);

            // Thread-safe addition for double using compare-exchange loop
            double currentSum, newSum;
            do
            {
                currentSum = _sum;
                newSum = currentSum + value;
            }
            while (Interlocked.CompareExchange(ref _sum, newSum, currentSum) != currentSum);

            // Update min/max (not perfectly atomic but acceptable for metrics)
            if (value < _min)
                _min = value;
            if (value > _max)
                _max = value;
        }

        /// <summary>
        /// Gets the number of recorded values.
        /// </summary>
        public long Count => Volatile.Read(ref _count);

        /// <summary>
        /// Gets the sum of all recorded values.
        /// </summary>
        public double Sum => Volatile.Read(ref _sum);

        /// <summary>
        /// Gets the average of all recorded values.
        /// </summary>
        public double Average => _count > 0 ? _sum / _count : 0;

        /// <summary>
        /// Gets the minimum recorded value.
        /// </summary>
        public double Min => _min == double.MaxValue ? 0 : _min;

        /// <summary>
        /// Gets the maximum recorded value.
        /// </summary>
        public double Max => _max == double.MinValue ? 0 : _max;
    }
}
