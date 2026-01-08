using MLFramework.Visualization.Configuration;

namespace MLFramework.Visualization;

/// <summary>
/// Configuration for the TensorBoardVisualizer
/// </summary>
public class VisualizerConfiguration
{
    /// <summary>
    /// Gets or sets the storage configuration
    /// </summary>
    public StorageConfiguration StorageConfig { get; set; }

    /// <summary>
    /// Gets or sets whether to enable asynchronous operations
    /// </summary>
    public bool EnableAsync { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable profiling
    /// </summary>
    public bool EnableProfiling { get; set; } = true;

    /// <summary>
    /// Gets or sets whether the visualizer is enabled
    /// </summary>
    public bool IsEnabled { get; set; } = true;

    /// <summary>
    /// Gets or sets the run name/tag for this visualization session
    /// </summary>
    public string RunName { get; set; } = "default";

    /// <summary>
    /// Gets or sets additional metadata for this run
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();

    /// <summary>
    /// Creates a new default configuration with the specified log directory
    /// </summary>
    /// <param name="logDirectory">Directory where logs will be stored</param>
    /// <returns>A new VisualizerConfiguration instance</returns>
    public static VisualizerConfiguration CreateDefault(string logDirectory)
    {
        return new VisualizerConfiguration
        {
            StorageConfig = new StorageConfiguration
            {
                BackendType = "file",
                LogDirectory = logDirectory,
                ConnectionString = logDirectory
            },
            EnableAsync = true,
            EnableProfiling = true,
            IsEnabled = true,
            RunName = "default"
        };
    }

    /// <summary>
    /// Validates the configuration
    /// </summary>
    public void Validate()
    {
        if (StorageConfig == null)
        {
            throw new InvalidOperationException("StorageConfig cannot be null");
        }

        if (string.IsNullOrWhiteSpace(StorageConfig.LogDirectory))
        {
            throw new InvalidOperationException("StorageConfig.LogDirectory cannot be null or empty");
        }
    }
}
