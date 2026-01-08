namespace MachineLearning.Visualization.Scalars;

/// <summary>
/// Interface for logging scalar metrics (loss, accuracy, learning rate, etc.)
/// </summary>
public interface IScalarLogger
{
    /// <summary>
    /// Logs a scalar value synchronously
    /// </summary>
    /// <param name="name">Name of the metric</param>
    /// <param name="value">Value of the metric</param>
    /// <param name="step">Training step (auto-incremented if -1)</param>
    void LogScalar(string name, float value, long step = -1);

    /// <summary>
    /// Logs a scalar value synchronously (double overload)
    /// </summary>
    /// <param name="name">Name of the metric</param>
    /// <param name="value">Value of the metric</param>
    /// <param name="step">Training step (auto-incremented if -1)</param>
    void LogScalar(string name, double value, long step = -1);

    /// <summary>
    /// Logs a scalar value asynchronously
    /// </summary>
    /// <param name="name">Name of the metric</param>
    /// <param name="value">Value of the metric</param>
    /// <param name="step">Training step (auto-incremented if -1)</param>
    /// <returns>Task that completes when the value is logged</returns>
    Task LogScalarAsync(string name, float value, long step = -1);

    /// <summary>
    /// Gets a scalar series by name
    /// </summary>
    /// <param name="name">Name of the metric</param>
    /// <returns>Scalar series or null if not found</returns>
    ScalarSeries? GetSeries(string name);

    /// <summary>
    /// Gets a scalar series by name asynchronously
    /// </summary>
    /// <param name="name">Name of the metric</param>
    /// <returns>Task returning scalar series or null if not found</returns>
    Task<ScalarSeries?> GetSeriesAsync(string name);

    /// <summary>
    /// Gets all scalar series
    /// </summary>
    /// <returns>All scalar series</returns>
    IEnumerable<ScalarSeries> GetAllSeries();

    /// <summary>
    /// Gets a smoothed version of a scalar series
    /// </summary>
    /// <param name="name">Name of the metric</param>
    /// <param name="windowSize">Size of the moving average window</param>
    /// <returns>Smoothed series or null if not found</returns>
    ScalarSeries? GetSmoothedSeries(string name, int windowSize);

    /// <summary>
    /// Gets the latest value for all metrics
    /// </summary>
    /// <returns>Dictionary mapping metric names to their latest values</returns>
    Dictionary<string, float> GetLatestValues();

    /// <summary>
    /// Tags the current run with metadata for comparison
    /// </summary>
    /// <param name="runName">Name of the run</param>
    /// <param name="tags">Metadata tags</param>
    void TagRun(string runName, Dictionary<string, string> tags);
}
