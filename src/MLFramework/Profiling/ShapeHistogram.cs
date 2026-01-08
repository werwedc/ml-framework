using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLFramework.Profiling
{
    /// <summary>
    /// Histogram tracking shape frequency distribution
    /// </summary>
    public class ShapeHistogram
    {
        /// <summary>
        /// String representation of shape -> count
        /// </summary>
        public Dictionary<string, int> BinCounts { get; }

        /// <summary>
        /// Total number of samples recorded
        /// </summary>
        public int TotalSamples { get; private set; }

        /// <summary>
        /// Number of unique shapes observed
        /// </summary>
        public int UniqueShapes => BinCounts.Count;

        /// <summary>
        /// Most commonly seen shape
        /// </summary>
        public int[]? MostCommonShape { get; private set; }

        /// <summary>
        /// Count of the most common shape
        /// </summary>
        public int MostCommonCount { get; private set; }

        public ShapeHistogram()
        {
            BinCounts = new Dictionary<string, int>();
            TotalSamples = 0;
            MostCommonCount = 0;
        }

        /// <summary>
        /// Add a shape sample to the histogram
        /// </summary>
        public void AddSample(int[] shape)
        {
            if (shape == null)
                throw new ArgumentNullException(nameof(shape));

            var shapeKey = ShapeToString(shape);

            if (BinCounts.ContainsKey(shapeKey))
            {
                BinCounts[shapeKey]++;
            }
            else
            {
                BinCounts[shapeKey] = 1;
            }

            TotalSamples++;

            // Update most common shape
            if (BinCounts[shapeKey] > MostCommonCount)
            {
                MostCommonCount = BinCounts[shapeKey];
                MostCommonShape = shape.ToArray();
            }
        }

        /// <summary>
        /// Get the frequency (0-1) of a specific shape
        /// </summary>
        public double GetFrequency(int[] shape)
        {
            if (shape == null)
                throw new ArgumentNullException(nameof(shape));

            if (TotalSamples == 0)
                return 0.0;

            var shapeKey = ShapeToString(shape);
            return BinCounts.ContainsKey(shapeKey) ? (double)BinCounts[shapeKey] / TotalSamples : 0.0;
        }

        /// <summary>
        /// Get the top N most common shapes
        /// </summary>
        public List<(int[] shape, int count)> GetTopShapes(int count)
        {
            return BinCounts
                .OrderByDescending(kvp => kvp.Value)
                .Take(count)
                .Select(kvp => (StringToShape(kvp.Key), kvp.Value))
                .ToList();
        }

        /// <summary>
        /// Get the probability (0-1) of a specific shape occurring
        /// </summary>
        public double GetProbability(int[] shape)
        {
            return GetFrequency(shape);
        }

        /// <summary>
        /// Generate a text report of the histogram
        /// </summary>
        public string ToReport()
        {
            var sb = new StringBuilder();
            sb.AppendLine("Shape Histogram Report");
            sb.AppendLine("======================");
            sb.AppendLine($"Total Samples: {TotalSamples}");
            sb.AppendLine($"Unique Shapes: {UniqueShapes}");
            sb.AppendLine();

            if (MostCommonShape != null)
            {
                sb.AppendLine($"Most Common Shape: [{string.Join(", ", MostCommonShape)}]");
                sb.AppendLine($"Frequency: {((double)MostCommonCount / TotalSamples):P2}");
                sb.AppendLine();
            }

            sb.AppendLine("Top 10 Shapes:");
            var topShapes = GetTopShapes(10);
            for (int i = 0; i < topShapes.Count; i++)
            {
                var (shape, count) = topShapes[i];
                var frequency = (double)count / TotalSamples;
                sb.AppendLine($"{i + 1}. [{string.Join(", ", shape)}] - {count} samples ({frequency:P2})");
            }

            return sb.ToString();
        }

        private string ShapeToString(int[] shape)
        {
            return string.Join(",", shape);
        }

        private int[] StringToShape(string str)
        {
            return str.Split(',').Select(int.Parse).ToArray();
        }
    }
}
