using System;

namespace MLFramework.Pipeline
{
    /// <summary>
    /// Metadata about a stored activation checkpoint
    /// </summary>
    public class CheckpointMetadata
    {
        /// <summary>
        /// Micro-batch index
        /// </summary>
        public int MicroBatchIndex { get; }

        /// <summary>
        /// Memory size in bytes
        /// </summary>
        public long MemorySize { get; }

        /// <summary>
        /// Timestamp when stored
        /// </summary>
        public DateTime Timestamp { get; }

        /// <summary>
        /// Shape of the activation tensor
        /// </summary>
        public long[] Shape { get; }

        public CheckpointMetadata(int microBatchIndex, long memorySize, long[] shape)
        {
            MicroBatchIndex = microBatchIndex;
            MemorySize = memorySize;
            Timestamp = DateTime.UtcNow;
            Shape = shape ?? throw new ArgumentNullException(nameof(shape));
        }
    }
}
