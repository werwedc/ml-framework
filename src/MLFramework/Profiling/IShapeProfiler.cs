namespace MLFramework.Profiling
{
    /// <summary>
    /// Interface for shape profiling functionality
    /// </summary>
    public interface IShapeProfiler
    {
        /// <summary>
        /// Record a shape sample for a tensor and operation
        /// </summary>
        void RecordShape(string tensorName, string opName, int[] shape);

        /// <summary>
        /// Get the histogram for a specific tensor
        /// </summary>
        ShapeHistogram? GetHistogram(string tensorName);

        /// <summary>
        /// Get the most common shapes for a tensor
        /// </summary>
        List<int[]> GetCommonShapes(string tensorName, int count);

        /// <summary>
        /// Get statistical analysis for a tensor
        /// </summary>
        ShapeStatistics? GetShapeStatistics(string tensorName);

        /// <summary>
        /// Clear profiling data for a specific tensor
        /// </summary>
        void Clear(string tensorName);

        /// <summary>
        /// Clear all profiling data
        /// </summary>
        void ClearAll();
    }
}
