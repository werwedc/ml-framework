using System;

namespace MLFramework.Amp
{
    /// <summary>
    /// Statistics for the dynamic loss scaler
    /// </summary>
    public class DynamicScalerStats
    {
        /// <summary>
        /// Gets the current scale factor
        /// </summary>
        public float CurrentScale { get; }

        /// <summary>
        /// Gets the total number of overflows
        /// </summary>
        public int TotalOverflows { get; }

        /// <summary>
        /// Gets the total number of iterations without overflow
        /// </summary>
        public int TotalSuccessfulIterations { get; }

        /// <summary>
        /// Gets the number of times the scale was increased
        /// </summary>
        public int ScaleIncreaseCount { get; }

        /// <summary>
        /// Gets the number of times the scale was decreased
        /// </summary>
        public int ScaleDecreaseCount { get; }

        /// <summary>
        /// Gets the minimum scale reached
        /// </summary>
        public float MinScaleReached { get; }

        /// <summary>
        /// Gets the maximum scale reached
        /// </summary>
        public float MaxScaleReached { get; }

        /// <summary>
        /// Gets the success rate (iterations without overflow / total iterations)
        /// </summary>
        public float SuccessRate
        {
            get
            {
                int totalIterations = TotalSuccessfulIterations + TotalOverflows;
                if (totalIterations == 0) return 1.0f;
                return (float)TotalSuccessfulIterations / totalIterations;
            }
        }

        /// <summary>
        /// Creates a new DynamicScalerStats
        /// </summary>
        public DynamicScalerStats(
            float currentScale,
            int totalOverflows,
            int totalSuccessfulIterations,
            int scaleIncreaseCount,
            int scaleDecreaseCount,
            float minScaleReached,
            float maxScaleReached)
        {
            CurrentScale = currentScale;
            TotalOverflows = totalOverflows;
            TotalSuccessfulIterations = totalSuccessfulIterations;
            ScaleIncreaseCount = scaleIncreaseCount;
            ScaleDecreaseCount = scaleDecreaseCount;
            MinScaleReached = minScaleReached;
            MaxScaleReached = maxScaleReached;
        }

        /// <summary>
        /// Returns a string representation of the statistics
        /// </summary>
        public override string ToString()
        {
            return $"DynamicScalerStats(" +
                   $"Scale: {CurrentScale:F2}, " +
                   $"Overflows: {TotalOverflows}, " +
                   $"SuccessIterations: {TotalSuccessfulIterations}, " +
                   $"ScaleIncreases: {ScaleIncreaseCount}, " +
                   $"ScaleDecreases: {ScaleDecreaseCount}, " +
                   $"MinScale: {MinScaleReached:F2}, " +
                   $"MaxScale: {MaxScaleReached:F2}, " +
                   $"SuccessRate: {SuccessRate:P2})";
        }
    }
}
