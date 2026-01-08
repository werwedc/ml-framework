using System;

namespace MLFramework.Data
{
    /// <summary>
    /// Statistics for tracking pool performance and behavior.
    /// </summary>
    public class PoolStatistics
    {
        private int _rentCount;
        private int _returnCount;
        private int _missCount;
        private int _discardCount;

        /// <summary>
        /// Gets the total number of rent operations performed.
        /// </summary>
        public int RentCount => _rentCount;

        /// <summary>
        /// Gets the total number of return operations performed.
        /// </summary>
        public int ReturnCount => _returnCount;

        /// <summary>
        /// Gets the number of times the pool was empty, requiring a new item to be created.
        /// </summary>
        public int MissCount => _missCount;

        /// <summary>
        /// Gets the number of items discarded due to reaching the maximum pool size.
        /// </summary>
        public int DiscardCount => _discardCount;

        /// <summary>
        /// Gets the hit rate, calculated as the percentage of requests served from the pool.
        /// </summary>
        public double HitRate => _rentCount > 0 ? (double)(_rentCount - _missCount) / _rentCount : 0.0;

        /// <summary>
        /// Increments the rent counter.
        /// </summary>
        internal void IncrementRent()
        {
            Interlocked.Increment(ref _rentCount);
        }

        /// <summary>
        /// Increments the return counter.
        /// </summary>
        internal void IncrementReturn()
        {
            Interlocked.Increment(ref _returnCount);
        }

        /// <summary>
        /// Increments the miss counter.
        /// </summary>
        internal void IncrementMiss()
        {
            Interlocked.Increment(ref _missCount);
        }

        /// <summary>
        /// Increments the discard counter.
        /// </summary>
        internal void IncrementDiscard()
        {
            Interlocked.Increment(ref _discardCount);
        }

        /// <summary>
        /// Resets all statistics counters to zero.
        /// </summary>
        public void Reset()
        {
            _rentCount = 0;
            _returnCount = 0;
            _missCount = 0;
            _discardCount = 0;
        }
    }
}
