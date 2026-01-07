using System;

namespace MLFramework.Profiling
{
    /// <summary>
    /// Represents a single shape sample recorded during profiling
    /// </summary>
    public class ShapeSample
    {
        /// <summary>
        /// The shape dimensions
        /// </summary>
        public int[] Shape { get; set; }

        /// <summary>
        /// Timestamp when the sample was recorded
        /// </summary>
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// Name of the tensor being profiled
        /// </summary>
        public string TensorName { get; set; }

        /// <summary>
        /// Name of the operation that produced this shape
        /// </summary>
        public string OperationName { get; set; }

        public ShapeSample(int[] shape, string tensorName, string operationName)
        {
            Shape = shape ?? throw new ArgumentNullException(nameof(shape));
            TensorName = tensorName ?? throw new ArgumentNullException(nameof(tensorName));
            OperationName = operationName ?? throw new ArgumentNullException(nameof(operationName));
            Timestamp = DateTime.UtcNow;
        }

        public override bool Equals(object obj)
        {
            if (obj is ShapeSample other)
            {
                return TensorName == other.TensorName &&
                       OperationName == other.OperationName &&
                       Shape.SequenceEqual(other.Shape);
            }
            return false;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(
                TensorName,
                OperationName,
                Shape.Aggregate(0, (hash, val) => HashCode.Combine(hash, val))
            );
        }
    }
}
