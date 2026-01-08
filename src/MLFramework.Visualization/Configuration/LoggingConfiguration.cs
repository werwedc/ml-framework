namespace MLFramework.Visualization.Configuration;

/// <summary>
/// Configuration for logging behavior
/// </summary>
public class LoggingConfiguration
{
    /// <summary>
    /// Enable scalar metric logging
    /// </summary>
    public bool LogScalars { get; set; } = true;

    /// <summary>
    /// Enable histogram logging
    /// </summary>
    public bool LogHistograms { get; set; } = true;

    /// <summary>
    /// Enable computational graph logging
    /// </summary>
    public bool LogGraphs { get; set; } = true;

    /// <summary>
    /// Enable hyperparameter logging
    /// </summary>
    public bool LogHyperparameters { get; set; } = true;

    /// <summary>
    /// Prefix for scalar logs
    /// </summary>
    public string ScalarLogPrefix { get; set; } = "";

    /// <summary>
    /// Number of bins in histogram
    /// </summary>
    public int HistogramBinCount { get; set; } = 30;

    /// <summary>
    /// Enable automatic smoothing of scalar values
    /// </summary>
    public bool AutoSmoothScalars { get; set; } = true;

    /// <summary>
    /// Default smoothing window size
    /// </summary>
    public int DefaultSmoothingWindow { get; set; } = 10;
}
