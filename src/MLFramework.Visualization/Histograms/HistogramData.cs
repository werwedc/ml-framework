namespace MLFramework.Visualization.Histograms;

/// <summary>
/// Represents histogram data with bin counts and statistics
/// </summary>
public class HistogramData
{
    /// <summary>
    /// Gets the name of the histogram
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the training step at which this histogram was recorded
    /// </summary>
    public long Step { get; }

    /// <summary>
    /// Gets the timestamp when this histogram was created
    /// </summary>
    public DateTime Timestamp { get; }

    /// <summary>
    /// Gets the bin edges (length = BinCount + 1)
    /// </summary>
    public float[] BinEdges { get; }

    /// <summary>
    /// Gets the count of values in each bin
    /// </summary>
    public long[] BinCounts { get; }

    /// <summary>
    /// Gets the minimum value in the data
    /// </summary>
    public float Min { get; }

    /// <summary>
    /// Gets the maximum value in the data
    /// </summary>
    public float Max { get; }

    /// <summary>
    /// Gets the mean value
    /// </summary>
    public float Mean { get; }

    /// <summary>
    /// Gets the standard deviation
    /// </summary>
    public float Std { get; }

    /// <summary>
    /// Gets the quantiles [0.1, 0.25, 0.5, 0.75, 0.9]
    /// </summary>
    public float[] Quantiles { get; }

    /// <summary>
    /// Gets the total count of values
    /// </summary>
    public int TotalCount { get; }

    /// <summary>
    /// Gets the average bin width
    /// </summary>
    public float BinWidth { get; }

    /// <summary>
    /// Private constructor for creating histogram data
    /// </summary>
    private HistogramData(
        string name,
        long step,
        DateTime timestamp,
        float[] binEdges,
        long[] binCounts,
        float min,
        float max,
        float mean,
        float std,
        float[] quantiles,
        int totalCount)
    {
        Name = name;
        Step = step;
        Timestamp = timestamp;
        BinEdges = binEdges;
        BinCounts = binCounts;
        Min = min;
        Max = max;
        Mean = mean;
        Std = std;
        Quantiles = quantiles;
        TotalCount = totalCount;
        BinWidth = binEdges.Length > 1 ? binEdges[1] - binEdges[0] : 0;
    }

    /// <summary>
    /// Creates a histogram from the given values with the specified configuration
    /// </summary>
    /// <param name="name">Name of the histogram</param>
    /// <param name="values">Array of values to histogram</param>
    /// <param name="config">Binning configuration</param>
    /// <param name="step">Training step</param>
    /// <returns>A new HistogramData instance</returns>
    public static HistogramData Create(string name, float[] values, HistogramBinConfig config, long step = -1)
    {
        if (string.IsNullOrEmpty(name))
        {
            throw new ArgumentException("Name cannot be null or empty", nameof(name));
        }

        if (values == null || values.Length == 0)
        {
            throw new ArgumentException("Values array cannot be null or empty", nameof(values));
        }

        if (config == null)
        {
            throw new ArgumentNullException(nameof(config));
        }

        config.Validate();

        // Determine the actual min/max for binning
        float dataMin = float.MaxValue;
        float dataMax = float.MinValue;

        for (int i = 0; i < values.Length; i++)
        {
            if (values[i] < dataMin) dataMin = values[i];
            if (values[i] > dataMax) dataMax = values[i];
        }

        float binMin = config.Min == float.MinValue ? dataMin : config.Min;
        float binMax = config.Max == float.MaxValue ? dataMax : config.Max;

        // Handle case where all values are the same
        if (binMin == binMax)
        {
            binMin -= 0.5f;
            binMax += 0.5f;
        }

        // Create bin edges
        float[] binEdges = CreateBinEdges(binMin, binMax, config.BinCount, config.UseLogScale);
        long[] binCounts = new long[config.BinCount];

        // Bin the values
        for (int i = 0; i < values.Length; i++)
        {
            int binIndex = FindBin(values[i], binEdges, config.UseLogScale);
            if (binIndex >= 0 && binIndex < binCounts.Length)
            {
                binCounts[binIndex]++;
            }
        }

        // Calculate statistics
        (float mean, float std) = CalculateMeanAndStd(values);
        float[] quantiles = Statistics.QuantileCalculator.CalculateMultiple(values, new[] { 0.1f, 0.25f, 0.5f, 0.75f, 0.9f });

        return new HistogramData(
            name,
            step,
            DateTime.UtcNow,
            binEdges,
            binCounts,
            dataMin,
            dataMax,
            mean,
            std,
            quantiles,
            values.Length);
    }

    /// <summary>
    /// Creates bin edges for the histogram
    /// </summary>
    private static float[] CreateBinEdges(float min, float max, int binCount, bool useLogScale)
    {
        var edges = new float[binCount + 1];

        if (useLogScale)
        {
            // Use logarithmic binning
            float logMin = (float)Math.Log(Math.Max(min, float.Epsilon));
            float logMax = (float)Math.Log(Math.Max(max, float.Epsilon));
            float logStep = (logMax - logMin) / binCount;

            for (int i = 0; i <= binCount; i++)
            {
                edges[i] = (float)Math.Exp(logMin + i * logStep);
            }
        }
        else
        {
            // Use linear binning
            float step = (max - min) / binCount;
            for (int i = 0; i <= binCount; i++)
            {
                edges[i] = min + i * step;
            }
        }

        return edges;
    }

    /// <summary>
    /// Finds the bin index for a given value
    /// </summary>
    private static int FindBin(float value, float[] binEdges, bool useLogScale)
    {
        if (value < binEdges[0] || value > binEdges[binEdges.Length - 1])
        {
            return -1; // Value outside bin range
        }

        if (useLogScale)
        {
            // Binary search for logarithmic bins
            int low = 0;
            int high = binEdges.Length - 2;

            while (low <= high)
            {
                int mid = (low + high) / 2;
                if (value >= binEdges[mid] && value < binEdges[mid + 1])
                {
                    return mid;
                }
                else if (value < binEdges[mid])
                {
                    high = mid - 1;
                }
                else
                {
                    low = mid + 1;
                }
            }

            // Handle the case where value is exactly at the last edge
            if (value == binEdges[binEdges.Length - 1])
            {
                return binEdges.Length - 2;
            }

            return -1;
        }
        else
        {
            // Linear bins can use simple calculation
            float binWidth = binEdges[1] - binEdges[0];
            int binIndex = (int)((value - binEdges[0]) / binWidth);

            // Clamp to valid range
            if (binIndex >= binEdges.Length - 1)
            {
                binIndex = binEdges.Length - 2;
            }

            return binIndex;
        }
    }

    /// <summary>
    /// Calculates mean and standard deviation in a single pass
    /// Uses Welford's algorithm for numerical stability
    /// </summary>
    private static (float Mean, float Std) CalculateMeanAndStd(float[] values)
    {
        int n = values.Length;

        if (n == 1)
        {
            return (values[0], 0);
        }

        float mean = 0f;
        float m2 = 0f; // Sum of squared differences from current mean

        // Welford's algorithm
        for (int i = 0; i < n; i++)
        {
            float delta = values[i] - mean;
            mean += delta / (i + 1);
            float delta2 = values[i] - mean;
            m2 += delta * delta2;
        }

        float variance = m2 / n;
        float std = (float)Math.Sqrt(variance);

        return (mean, std);
    }
}
