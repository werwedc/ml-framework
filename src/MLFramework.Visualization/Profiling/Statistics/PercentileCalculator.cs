namespace MLFramework.Visualization.Profiling.Statistics;

/// <summary>
/// Calculates percentiles from duration data
/// </summary>
public class PercentileCalculator
{
    /// <summary>
    /// Calculates the median (50th percentile)
    /// </summary>
    /// <param name="values">Array of values</param>
    /// <returns>Median value</returns>
    public static long CalculateMedian(long[] values)
    {
        if (values == null || values.Length == 0)
        {
            return 0;
        }

        // Make a copy to avoid modifying the original array
        var sorted = new long[values.Length];
        Array.Copy(values, sorted, values.Length);
        Array.Sort(sorted);

        int n = sorted.Length;

        if (n % 2 == 0)
        {
            // Even number of elements - take average of middle two
            return (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
        }
        else
        {
            // Odd number of elements - take middle element
            return sorted[n / 2];
        }
    }

    /// <summary>
    /// Calculates a specific percentile
    /// </summary>
    /// <param name="values">Array of values</param>
    /// <param name="percentile">Percentile to calculate (0-100)</param>
    /// <returns>Value at the specified percentile</returns>
    public static long CalculatePercentile(long[] values, double percentile)
    {
        if (values == null || values.Length == 0)
        {
            return 0;
        }

        if (percentile <= 0)
        {
            return values.Min();
        }

        if (percentile >= 100)
        {
            return values.Max();
        }

        // Make a copy to avoid modifying the original array
        var sorted = new long[values.Length];
        Array.Copy(values, sorted, values.Length);
        Array.Sort(sorted);

        int n = sorted.Length;
        double position = (percentile / 100.0) * (n - 1);
        int lowerIndex = (int)Math.Floor(position);
        int upperIndex = (int)Math.Ceiling(position);

        if (lowerIndex == upperIndex)
        {
            return sorted[lowerIndex];
        }

        // Interpolate between the two values
        double weight = position - lowerIndex;
        return (long)(sorted[lowerIndex] * (1 - weight) + sorted[upperIndex] * weight);
    }

    /// <summary>
    /// Calculates multiple percentiles efficiently
    /// </summary>
    /// <param name="values">Array of values</param>
    /// <param name="percentiles">Array of percentiles to calculate (0-100)</param>
    /// <returns>Array of percentile values in the same order as input</returns>
    public static long[] CalculatePercentiles(long[] values, double[] percentiles)
    {
        if (values == null || values.Length == 0)
        {
            return new long[percentiles.Length];
        }

        if (percentiles == null || percentiles.Length == 0)
        {
            return Array.Empty<long>();
        }

        // Sort values once
        var sorted = new long[values.Length];
        Array.Copy(values, sorted, values.Length);
        Array.Sort(sorted);

        int n = sorted.Length;
        var results = new long[percentiles.Length];

        for (int i = 0; i < percentiles.Length; i++)
        {
            double percentile = percentiles[i];

            if (percentile <= 0)
            {
                results[i] = sorted[0];
            }
            else if (percentile >= 100)
            {
                results[i] = sorted[n - 1];
            }
            else
            {
                double position = (percentile / 100.0) * (n - 1);
                int lowerIndex = (int)Math.Floor(position);
                int upperIndex = (int)Math.Ceiling(position);

                if (lowerIndex == upperIndex)
                {
                    results[i] = sorted[lowerIndex];
                }
                else
                {
                    double weight = position - lowerIndex;
                    results[i] = (long)(sorted[lowerIndex] * (1 - weight) + sorted[upperIndex] * weight);
                }
            }
        }

        return results;
    }

    /// <summary>
    /// Calculates common percentiles (50, 90, 95, 99)
    /// </summary>
    /// <param name="values">Array of values</param>
    /// <returns>Tuple containing p50, p90, p95, p99</returns>
    public static (long P50, long P90, long P95, long P99) CalculateCommonPercentiles(long[] values)
    {
        if (values == null || values.Length == 0)
        {
            return (0, 0, 0, 0);
        }

        var percentiles = CalculatePercentiles(values, new[] { 50.0, 90.0, 95.0, 99.0 });
        return (percentiles[0], percentiles[1], percentiles[2], percentiles[3]);
    }
}
