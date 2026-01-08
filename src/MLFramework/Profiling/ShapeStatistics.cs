using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Profiling
{
    /// <summary>
    /// Statistical analysis of shape distributions
    /// </summary>
    public class ShapeStatistics
    {
        /// <summary>
        /// Mean shape dimensions (averaged across samples)
        /// </summary>
        public double[] MeanShape { get; private set; }

        /// <summary>
        /// Standard deviation of each dimension
        /// </summary>
        public double[] StdDevShape { get; private set; }

        /// <summary>
        /// Minimum shape observed
        /// </summary>
        public int[] MinShape { get; private set; }

        /// <summary>
        /// Maximum shape observed
        /// </summary>
        public int[] MaxShape { get; private set; }

        /// <summary>
        /// Percentiles for each dimension (e.g., 25th, 50th, 75th)
        /// </summary>
        public Dictionary<int, double[]> Percentiles { get; private set; }

        public ShapeStatistics()
        {
            Percentiles = new Dictionary<int, double[]>();
        }

        /// <summary>
        /// Calculate statistics from a shape histogram
        /// </summary>
        public void CalculateFromHistogram(ShapeHistogram histogram)
        {
            if (histogram == null)
                throw new ArgumentNullException(nameof(histogram));

            if (histogram.TotalSamples == 0)
                return;

            // Collect all shapes with their weights
            var allShapes = new List<(int[] shape, int count)>();
            foreach (var kvp in histogram.BinCounts)
            {
                var shape = kvp.Key.Split(',').Select(int.Parse).ToArray();
                allShapes.Add((shape, kvp.Value));
            }

            if (allShapes.Count == 0)
                return;

            int dimensions = allShapes[0].shape.Length;
            int totalSamples = histogram.TotalSamples;

            // Calculate mean
            MeanShape = new double[dimensions];
            foreach (var (shape, count) in allShapes)
            {
                for (int i = 0; i < dimensions; i++)
                {
                    MeanShape[i] += shape[i] * count;
                }
            }
            for (int i = 0; i < dimensions; i++)
            {
                MeanShape[i] /= totalSamples;
            }

            // Calculate standard deviation
            StdDevShape = new double[dimensions];
            foreach (var (shape, count) in allShapes)
            {
                for (int i = 0; i < dimensions; i++)
                {
                    double diff = shape[i] - MeanShape[i];
                    StdDevShape[i] += diff * diff * count;
                }
            }
            for (int i = 0; i < dimensions; i++)
            {
                StdDevShape[i] = Math.Sqrt(StdDevShape[i] / totalSamples);
            }

            // Calculate min and max
            MinShape = new int[dimensions];
            MaxShape = new int[dimensions];
            for (int i = 0; i < dimensions; i++)
            {
                MinShape[i] = int.MaxValue;
                MaxShape[i] = int.MinValue;
            }

            foreach (var (shape, count) in allShapes)
            {
                for (int i = 0; i < dimensions; i++)
                {
                    if (shape[i] < MinShape[i]) MinShape[i] = shape[i];
                    if (shape[i] > MaxShape[i]) MaxShape[i] = shape[i];
                }
            }

            // Calculate percentiles (25th, 50th, 75th)
            Percentiles[25] = new double[dimensions];
            Percentiles[50] = new double[dimensions];
            Percentiles[75] = new double[dimensions];

            for (int dim = 0; dim < dimensions; dim++)
            {
                // Collect all values for this dimension
                var values = new List<(int value, int count)>();
                foreach (var (shape, count) in allShapes)
                {
                    values.Add((shape[dim], count));
                }

                Percentiles[50][dim] = CalculatePercentile(values, 0.5, totalSamples);
                Percentiles[25][dim] = CalculatePercentile(values, 0.25, totalSamples);
                Percentiles[75][dim] = CalculatePercentile(values, 0.75, totalSamples);
            }
        }

        private double CalculatePercentile(List<(int value, int count)> values, double percentile, int totalSamples)
        {
            values.Sort((a, b) => a.value.CompareTo(b.value));

            int targetIndex = (int)Math.Ceiling(totalSamples * percentile) - 1;
            targetIndex = Math.Max(0, Math.Min(targetIndex, totalSamples - 1));

            int cumulativeCount = 0;
            foreach (var (value, count) in values)
            {
                cumulativeCount += count;
                if (cumulativeCount > targetIndex)
                {
                    return value;
                }
            }

            return values.Last().value;
        }
    }
}
