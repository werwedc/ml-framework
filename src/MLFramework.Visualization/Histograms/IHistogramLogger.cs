namespace MLFramework.Visualization.Histograms;

/// <summary>
/// Configuration for histogram binning
/// </summary>
public class HistogramBinConfig
{
    /// <summary>
    /// Number of bins in the histogram (default: 30)
    /// </summary>
    public int BinCount { get; set; } = 30;

    /// <summary>
    /// Whether to use logarithmic scale for bins (default: false)
    /// </summary>
    public bool UseLogScale { get; set; } = false;

    /// <summary>
    /// Minimum value for binning (default: auto-detect)
    /// </summary>
    public float Min { get; set; } = float.MinValue;

    /// <summary>
    /// Maximum value for binning (default: auto-detect)
    /// </summary>
    public float Max { get; set; } = float.MaxValue;

    /// <summary>
    /// Validates the configuration
    /// </summary>
    public void Validate()
    {
        if (BinCount <= 0)
        {
            throw new ArgumentException("BinCount must be greater than 0", nameof(BinCount));
        }

        if (Min >= Max)
        {
            throw new ArgumentException("Min must be less than Max", nameof(Min));
        }
    }
}

/// <summary>
/// Interface for logging histogram data (weight/gradient distributions, etc.)
/// </summary>
public interface IHistogramLogger
{
    /// <summary>
    /// Logs a histogram synchronously
    /// </summary>
    /// <param name="name">Name of the histogram</param>
    /// <param name="values">Array of histogram values</param>
    /// <param name="step">Training step (auto-incremented if -1)</param>
    void LogHistogram(string name, float[] values, long step = -1);

    /// <summary>
    /// Logs a histogram synchronously with custom binning configuration
    /// </summary>
    /// <param name="name">Name of the histogram</param>
    /// <param name="values">Array of histogram values</param>
    /// <param name="config">Binning configuration</param>
    /// <param name="step">Training step (auto-incremented if -1)</param>
    void LogHistogram(string name, float[] values, HistogramBinConfig config, long step = -1);

    /// <summary>
    /// Logs a histogram asynchronously
    /// </summary>
    /// <param name="name">Name of the histogram</param>
    /// <param name="values">Array of histogram values</param>
    /// <param name="step">Training step (auto-incremented if -1)</param>
    /// <returns>Task that completes when the histogram is logged</returns>
    Task LogHistogramAsync(string name, float[] values, long step = -1);

    /// <summary>
    /// Logs a histogram asynchronously with custom binning configuration
    /// </summary>
    /// <param name="name">Name of the histogram</param>
    /// <param name="values">Array of histogram values</param>
    /// <param name="config">Binning configuration</param>
    /// <param name="step">Training step (auto-incremented if -1)</param>
    /// <returns>Task that completes when the histogram is logged</returns>
    Task LogHistogramAsync(string name, float[] values, HistogramBinConfig config, long step = -1);
}
