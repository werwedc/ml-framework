namespace MLFramework.Data
{
    /// <summary>
    /// Statistics for monitoring the behavior of a SharedQueue.
    /// </summary>
    public class QueueStatistics
    {
        /// <summary>
        /// Gets or sets the total number of items enqueued.
        /// </summary>
        public int TotalEnqueued { get; set; }

        /// <summary>
        /// Gets or sets the total number of items dequeued.
        /// </summary>
        public int TotalDequeued { get; set; }

        /// <summary>
        /// Gets or sets the average wait time in milliseconds for producers and consumers.
        /// </summary>
        public long AverageWaitTimeMs { get; set; }

        /// <summary>
        /// Gets or sets the maximum queue size observed.
        /// </summary>
        public int MaxQueueSize { get; set; }

        /// <summary>
        /// Gets or sets the number of times producers had to wait for space.
        /// </summary>
        public int ProducerWaitCount { get; set; }

        /// <summary>
        /// Gets or sets the number of times consumers had to wait for items.
        /// </summary>
        public int ConsumerWaitCount { get; set; }

        // Internal fields for tracking statistics
        internal long TotalWaitTimeMs { get; set; }
        internal int TotalWaitCount { get; set; }

        /// <summary>
        /// Creates a new instance of QueueStatistics with default values.
        /// </summary>
        public QueueStatistics()
        {
        }

        /// <summary>
        /// Creates a copy of the current statistics.
        /// </summary>
        /// <returns>A new QueueStatistics instance with the same values.</returns>
        public QueueStatistics Clone()
        {
            return new QueueStatistics
            {
                TotalEnqueued = TotalEnqueued,
                TotalDequeued = TotalDequeued,
                AverageWaitTimeMs = AverageWaitTimeMs,
                MaxQueueSize = MaxQueueSize,
                ProducerWaitCount = ProducerWaitCount,
                ConsumerWaitCount = ConsumerWaitCount,
                TotalWaitTimeMs = TotalWaitTimeMs,
                TotalWaitCount = TotalWaitCount
            };
        }

        /// <summary>
        /// Returns a string representation of the statistics.
        /// </summary>
        /// <returns>A formatted string with statistics information.</returns>
        public override string ToString()
        {
            return $"QueueStatistics: " +
                   $"Enqueued={TotalEnqueued}, " +
                   $"Dequeued={TotalDequeued}, " +
                   $"AvgWait={AverageWaitTimeMs}ms, " +
                   $"MaxSize={MaxQueueSize}, " +
                   $"ProducerWaits={ProducerWaitCount}, " +
                   $"ConsumerWaits={ConsumerWaitCount}";
        }
    }
}
