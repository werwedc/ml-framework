namespace MLFramework.Amp
{
    /// <summary>
    /// Performance statistics for a GPU kernel
    /// </summary>
    public class KernelPerformanceStats
    {
        /// <summary>
        /// Gets the operation name
        /// </summary>
        public string OperationName { get; }

        /// <summary>
        /// Gets the kernel data type
        /// </summary>
        public KernelDtype Dtype { get; }

        /// <summary>
        /// Gets the average execution time (ms)
        /// </summary>
        public float AverageExecutionTime { get; }

        /// <summary>
        /// Gets the minimum execution time (ms)
        /// </summary>
        public float MinExecutionTime { get; }

        /// <summary>
        /// Gets the maximum execution time (ms)
        /// </summary>
        public float MaxExecutionTime { get; }

        /// <summary>
        /// Gets the number of executions
        /// </summary>
        public int ExecutionCount { get; }

        /// <summary>
        /// Creates a new KernelPerformanceStats
        /// </summary>
        public KernelPerformanceStats(
            string operationName,
            KernelDtype dtype,
            float averageExecutionTime,
            float minExecutionTime,
            float maxExecutionTime,
            int executionCount)
        {
            OperationName = operationName;
            Dtype = dtype;
            AverageExecutionTime = averageExecutionTime;
            MinExecutionTime = minExecutionTime;
            MaxExecutionTime = maxExecutionTime;
            ExecutionCount = executionCount;
        }

        /// <summary>
        /// Returns a string representation of the statistics
        /// </summary>
        public override string ToString()
        {
            return $"KernelPerformanceStats(" +
                   $"Operation={OperationName}, " +
                   $"Dtype={Dtype}, " +
                   $"AvgTime={AverageExecutionTime:F3}ms, " +
                   $"MinTime={MinExecutionTime:F3}ms, " +
                   $"MaxTime={MaxExecutionTime:F3}ms, " +
                   $"Executions={ExecutionCount})";
        }
    }
}
