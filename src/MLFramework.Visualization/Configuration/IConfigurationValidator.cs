using System.Text.Json;

namespace MLFramework.Visualization.Configuration;

/// <summary>
/// Interface for configuration validation operations
/// </summary>
public interface IConfigurationValidator
{
    /// <summary>
    /// Validate configuration and return result
    /// </summary>
    ValidationResult Validate(VisualizationConfiguration config);

    /// <summary>
    /// Validate configuration and throw exception if invalid
    /// </summary>
    void ValidateAndThrow(VisualizationConfiguration config);
}
