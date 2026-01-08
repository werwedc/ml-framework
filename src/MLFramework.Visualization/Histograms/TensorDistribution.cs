using MLFramework.Visualization.Histograms.Statistics;

namespace MLFramework.Visualization.Histograms;

/// <summary>
/// Represents a tensor distribution with extended statistics and histogram
/// </summary>
public class TensorDistribution
{
    /// <summary>
    /// Gets the name of the tensor distribution
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the training step at which this distribution was recorded
    /// </summary>
    public long Step { get; }

    /// <summary>
    /// Gets the tensor values
    /// </summary>
    public float[] Values { get; }

    /// <summary>
    /// Gets the histogram for this distribution
    /// </summary>
    public HistogramData Histogram { get; }

    /// <summary>
    /// Gets the median value
    /// </summary>
    public float Median { get; }

    /// <summary>
    /// Gets the skewness (third standardized moment)
    /// </summary>
    public float Skewness { get; }

    /// <summary>
    /// Gets the kurtosis (fourth standardized moment, excess kurtosis)
    /// </summary>
    public float Kurtosis { get; }

    /// <summary>
    /// Gets the count of dead neurons (zero-valued entries)
    /// </summary>
    public int DeadNeuronCount { get; }

    /// <summary>
    /// Gets the count of outliers (values > 3 standard deviations from mean)
    /// </summary>
    public int OutlierCount { get; }

    /// <summary>
    /// Gets the timestamp when this distribution was created
    /// </summary>
    public DateTime Timestamp { get; }

    /// <summary>
    /// Gets the mean value
    /// </summary>
    public float Mean => Histogram.Mean;

    /// <summary>
    /// Gets the standard deviation
    /// </summary>
    public float Std => Histogram.Std;

    /// <summary>
    /// Gets the minimum value
    /// </summary>
    public float Min => Histogram.Min;

    /// <summary>
    /// Gets the maximum value
    /// </summary>
    public float Max => Histogram.Max;

    /// <summary>
    /// Gets the quantiles
    /// </summary>
    public float[] Quantiles => Histogram.Quantiles;

    /// <summary>
    /// Gets the total count of values
    /// </summary>
    public int TotalCount => Histogram.TotalCount;

    /// <summary>
    /// Private constructor for creating tensor distribution data
    /// </summary>
    private TensorDistribution(
        string name,
        long step,
        float[] values,
        HistogramData histogram,
        float median,
        float skewness,
        float kurtosis,
        int deadNeuronCount,
        int outlierCount)
    {
        Name = name;
        Step = step;
        Values = values;
        Histogram = histogram;
        Median = median;
        Skewness = skewness;
        Kurtosis = kurtosis;
        DeadNeuronCount = deadNeuronCount;
        OutlierCount = outlierCount;
        Timestamp = DateTime.UtcNow;
    }

    /// <summary>
    /// Creates a tensor distribution from the given values with the specified configuration
    /// </summary>
    /// <param name="name">Name of the tensor distribution</param>
    /// <param name="values">Array of values to analyze</param>
    /// <param name="config">Binning configuration for the histogram</param>
    /// <param name="step">Training step</param>
    /// <param name="detectOutliers">Whether to detect outliers</param>
    /// <param name="outlierThresholdStd">Number of standard deviations for outlier detection</param>
    /// <returns>A new TensorDistribution instance</returns>
    public static TensorDistribution Create(
        string name,
        float[] values,
        HistogramBinConfig config,
        long step = -1,
        bool detectOutliers = true,
        float outlierThresholdStd = 3.0f)
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

        // Create histogram
        var histogram = HistogramData.Create(name, values, config, step);

        // Calculate median
        float median = QuantileCalculator.Calculate(values, 0.5f);

        // Calculate skewness and kurtosis
        (float skewness, float kurtosis) = MomentsCalculator.CalculateBoth(values);

        // Count dead neurons (zeros)
        int deadNeuronCount = OutlierDetector.CountDeadNeurons(values);

        // Count outliers
        int outlierCount = 0;
        if (detectOutliers)
        {
            (outlierCount, _) = OutlierDetector.DetectOutliersByStd(values, outlierThresholdStd);
        }

        return new TensorDistribution(
            name,
            step,
            values,
            histogram,
            median,
            skewness,
            kurtosis,
            deadNeuronCount,
            outlierCount);
    }

    /// <summary>
    /// Creates a tensor distribution with default binning configuration
    /// </summary>
    /// <param name="name">Name of the tensor distribution</param>
    /// <param name="values">Array of values to analyze</param>
    /// <param name="step">Training step</param>
    /// <returns>A new TensorDistribution instance</returns>
    public static TensorDistribution Create(string name, float[] values, long step = -1)
    {
        return Create(name, values, new HistogramBinConfig(), step);
    }

    /// <summary>
    /// Gets the percentage of dead neurons
    /// </summary>
    public float DeadNeuronPercentage => TotalCount > 0 ? (float)DeadNeuronCount / TotalCount * 100 : 0;

    /// <summary>
    /// Gets the percentage of outliers
    /// </summary>
    public float OutlierPercentage => TotalCount > 0 ? (float)OutlierCount / TotalCount * 100 : 0;

    /// <summary>
    /// Checks if the distribution has a significant number of dead neurons (> 10%)
    /// </summary>
    public bool HasDeadNeuronIssue => DeadNeuronPercentage > 10.0f;

    /// <summary>
    /// Checks if the distribution has a significant number of outliers (> 5%)
    /// </summary>
    public bool HasOutlierIssue => OutlierPercentage > 5.0f;

    /// <summary>
    /// Gets a summary of the distribution
    /// </summary>
    public string GetSummary()
    {
        return $"TensorDistribution '{Name}' at step {Step}:\n" +
               $"  Count: {TotalCount}\n" +
               $"  Mean: {Mean:F6}, Std: {Std:F6}\n" +
               $"  Min: {Min:F6}, Max: {Max:F6}\n" +
               $"  Median: {Median:F6}\n" +
               $"  Skewness: {Skewness:F6}, Kurtosis: {Kurtosis:F6}\n" +
               $"  Dead Neurons: {DeadNeuronCount} ({DeadNeuronPercentage:F2}%)\n" +
               $"  Outliers: {OutlierCount} ({OutlierPercentage:F2}%)\n" +
               $"  Quantiles [10%, 25%, 50%, 75%, 90%]: [{Quantiles[0]:F6}, {Quantiles[1]:F6}, {Quantiles[2]:F6}, {Quantiles[3]:F6}, {Quantiles[4]:F6}]";
    }
}
