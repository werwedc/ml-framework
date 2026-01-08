using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Profiling
{
    /// <summary>
    /// Provides optimization recommendations based on shape profiling data
    /// </summary>
    public class ShapeProfileOptimizer
    {
        /// <summary>
        /// Recommend shapes that are worth specializing based on frequency threshold
        /// </summary>
        public List<int[]> RecommendSpecializedShapes(string tensorName, ShapeHistogram histogram, int threshold)
        {
            if (histogram == null)
                throw new ArgumentNullException(nameof(histogram));

            if (histogram.TotalSamples == 0)
                return new List<int[]>();

            // Filter shapes that appear at least threshold times
            var result = new List<int[]>();
            foreach (var kvp in histogram.BinCounts)
            {
                if (kvp.Value >= threshold)
                {
                    var shape = kvp.Key.Split(',').Select(int.Parse).ToArray();
                    result.Add(shape);
                }
            }

            // Sort by frequency (descending)
            result.Sort((a, b) =>
            {
                var freqA = histogram.GetFrequency(a);
                var freqB = histogram.GetFrequency(b);
                return freqB.CompareTo(freqA);
            });

            return result;
        }

        /// <summary>
        /// Determine if recompilation is recommended based on new shape and histogram
        /// </summary>
        public bool ShouldRecompile(int[] newShape, ShapeHistogram histogram, double threshold = 0.05)
        {
            if (newShape == null)
                throw new ArgumentNullException(nameof(newShape));

            if (histogram == null)
                throw new ArgumentNullException(nameof(histogram));

            if (histogram.TotalSamples == 0)
                return true;

            // Check if this shape is already common
            var frequency = histogram.GetFrequency(newShape);

            // If the shape is already common (above threshold), no need to recompile
            // If it's rare, we might want to recompile to optimize for it
            return frequency < threshold;
        }

        /// <summary>
        /// Get optimal padding bounds based on histogram distribution
        /// </summary>
        public int[] GetOptimalPadding(ShapeHistogram histogram)
        {
            if (histogram == null)
                throw new ArgumentNullException(nameof(histogram));

            if (histogram.TotalSamples == 0)
                return Array.Empty<int>();

            // Get the most common shape as a baseline
            if (histogram.MostCommonShape == null)
                return Array.Empty<int>();

            var mostCommon = histogram.MostCommonShape;
            var topShapes = histogram.GetTopShapes(10);

            if (topShapes.Count == 0)
                return mostCommon;

            // Calculate padding that covers the top shapes efficiently
            int dimensions = mostCommon.Length;
            var padding = new int[dimensions];

            for (int dim = 0; dim < dimensions; dim++)
            {
                // Find the maximum value in this dimension among top shapes
                int maxDim = mostCommon[dim];
                foreach (var (shape, _) in topShapes)
                {
                    if (shape.Length > dim && shape[dim] > maxDim)
                    {
                        maxDim = shape[dim];
                    }
                }

                // Round up to the nearest power of 2 for alignment
                padding[dim] = RoundUpToPowerOf2(maxDim);
            }

            return padding;
        }

        /// <summary>
        /// Get the most efficient batch size based on histogram
        /// </summary>
        public int GetOptimalBatchSize(ShapeHistogram histogram, int batchSizeIndex = 0)
        {
            if (histogram == null)
                throw new ArgumentNullException(nameof(histogram));

            if (histogram.TotalSamples == 0)
                return 32; // Default batch size

            var topShapes = histogram.GetTopShapes(5);
            if (topShapes.Count == 0)
                return 32;

            // Assume the last dimension is typically batch size
            var batchSizes = topShapes
                .Where(x => x.shape.Length > 0)
                .Select(x => x.shape[batchSizeIndex])
                .Distinct()
                .OrderByDescending(x => x)
                .ToList();

            if (batchSizes.Count == 0)
                return 32;

            // Return the most common batch size (or largest for efficiency)
            return batchSizes[0];
        }

        /// <summary>
        /// Recommend whether to enable dynamic shape handling
        /// </summary>
        public bool RecommendDynamicShapes(ShapeHistogram histogram, double diversityThreshold = 0.3)
        {
            if (histogram == null)
                throw new ArgumentNullException(nameof(histogram));

            if (histogram.TotalSamples == 0)
                return true;

            // Calculate shape diversity (entropy-like measure)
            double diversity = 0.0;
            foreach (var kvp in histogram.BinCounts)
            {
                double freq = (double)kvp.Value / histogram.TotalSamples;
                diversity -= freq * Math.Log(freq);
            }

            // Maximum entropy for n unique shapes is log(n)
            double maxDiversity = Math.Log(histogram.UniqueShapes);
            double normalizedDiversity = maxDiversity > 0 ? diversity / maxDiversity : 0.0;

            return normalizedDiversity > diversityThreshold;
        }

        private int RoundUpToPowerOf2(int n)
        {
            if (n <= 0)
                return 1;

            n--;
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            n++;

            return n;
        }
    }
}
