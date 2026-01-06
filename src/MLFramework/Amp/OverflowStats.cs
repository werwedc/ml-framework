using System;
using System.Collections.Generic;
using System.Text;

namespace MLFramework.Amp
{
    /// <summary>
    /// Statistics about gradient overflow
    /// </summary>
    public class OverflowStats
    {
        /// <summary>
        /// Gets the total number of gradients checked
        /// </summary>
        public int TotalGradients { get; }

        /// <summary>
        /// Gets the number of gradients with overflow
        /// </summary>
        public int OverflowCount { get; }

        /// <summary>
        /// Gets the list of parameter names with overflow
        /// </summary>
        public IReadOnlyList<string> OverflowParameters { get; }

        /// <summary>
        /// Gets the overflow rate (overflow count / total count)
        /// </summary>
        public float OverflowRate { get; }

        /// <summary>
        /// Gets whether any overflow was detected
        /// </summary>
        public bool HasOverflow => OverflowCount > 0;

        /// <summary>
        /// Creates a new OverflowStats
        /// </summary>
        public OverflowStats(
            int totalGradients,
            int overflowCount,
            IReadOnlyList<string> overflowParameters)
        {
            if (totalGradients < 0)
                throw new ArgumentException("Total gradients must be non-negative", nameof(totalGradients));

            if (overflowCount < 0)
                throw new ArgumentException("Overflow count must be non-negative", nameof(overflowCount));

            if (overflowCount > totalGradients)
                throw new ArgumentException("Overflow count cannot exceed total gradients");

            if (overflowParameters == null)
                throw new ArgumentNullException(nameof(overflowParameters));

            TotalGradients = totalGradients;
            OverflowCount = overflowCount;
            OverflowParameters = overflowParameters;

            OverflowRate = totalGradients > 0 ? (float)overflowCount / totalGradients : 0.0f;
        }

        /// <summary>
        /// Returns a string representation of the statistics
        /// </summary>
        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.AppendLine($"Overflow Statistics:");
            sb.AppendLine($"  Total Gradients: {TotalGradients}");
            sb.AppendLine($"  Overflow Count: {OverflowCount}");
            sb.AppendLine($"  Overflow Rate: {OverflowRate:P2}");

            if (HasOverflow)
            {
                sb.AppendLine($"  Overflow Parameters:");
                foreach (var param in OverflowParameters)
                {
                    sb.AppendLine($"    - {param}");
                }
            }

            return sb.ToString().Trim();
        }
    }
}
