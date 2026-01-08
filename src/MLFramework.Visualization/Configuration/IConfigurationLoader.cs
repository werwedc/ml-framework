namespace MLFramework.Visualization.Configuration;

/// <summary>
/// Interface for configuration loading operations
/// </summary>
public interface IConfigurationLoader
{
    /// <summary>
    /// Load default configuration with environment variable overrides
    /// </summary>
    VisualizationConfiguration Load();

    /// <summary>
    /// Load configuration from a JSON file
    /// </summary>
    VisualizationConfiguration LoadFromFile(string filePath);

    /// <summary>
    /// Load configuration from a JSON string
    /// </summary>
    VisualizationConfiguration LoadFromJson(string json);

    /// <summary>
    /// Load configuration from environment variables
    /// </summary>
    VisualizationConfiguration LoadFromEnvironment();

    /// <summary>
    /// Save configuration to a JSON file
    /// </summary>
    void Save(VisualizationConfiguration config, string filePath);

    /// <summary>
    /// Save configuration to a JSON string
    /// </summary>
    string SaveToJson(VisualizationConfiguration config);
}
