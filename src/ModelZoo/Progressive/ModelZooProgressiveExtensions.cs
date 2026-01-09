using MLFramework.Core;
using MLFramework.ModelVersioning;

namespace MLFramework.ModelZoo.Progressive;

/// <summary>
/// Extension methods for adding progressive loading support to ModelZoo.
/// </summary>
public static class ModelZooProgressiveExtensions
{
    /// <summary>
    /// Loads a model progressively from the ModelZoo.
    /// </summary>
    /// <param name="modelZoo">The ModelZoo instance.</param>
    /// <param name="modelName">The model name.</param>
    /// <param name="version">The model version (optional).</param>
    /// <param name="device">The target device (optional).</param>
    /// <param name="options">Loading options (optional).</param>
    /// <returns>A ProgressiveModelLoader instance.</returns>
    public static ProgressiveModelLoader LoadProgressive(
        this ModelZoo modelZoo,
        string modelName,
        string? version = null,
        Device? device = null,
        ProgressiveLoadOptions? options = null)
    {
        if (modelZoo == null)
        {
            throw new ArgumentNullException(nameof(modelZoo));
        }

        return ProgressiveModelLoader.LoadProgressive(modelName, version, device, options);
    }

    /// <summary>
    /// Loads a model progressively with a specific loading strategy.
    /// </summary>
    /// <param name="modelZoo">The ModelZoo instance.</param>
    /// <param name="modelName">The model name.</param>
    /// <param name="version">The model version (optional).</param>
    /// <param name="strategy">The loading strategy to use.</param>
    /// <param name="device">The target device (optional).</param>
    /// <returns>A ProgressiveModelLoader instance.</returns>
    public static ProgressiveModelLoader LoadProgressive(
        this ModelZoo modelZoo,
        string modelName,
        LayerLoadingStrategy strategy,
        string? version = null,
        Device? device = null)
    {
        if (modelZoo == null)
        {
            throw new ArgumentNullException(nameof(modelZoo));
        }

        var options = new ProgressiveLoadOptions(strategy);
        return ProgressiveModelLoader.LoadProgressive(modelName, version, device, options);
    }

    /// <summary>
    /// Loads a model progressively from metadata.
    /// </summary>
    /// <param name="modelZoo">The ModelZoo instance.</param>
    /// <param name="metadata">The model metadata.</param>
    /// <param name="device">The target device (optional).</param>
    /// <param name="options">Loading options (optional).</param>
    /// <returns>A ProgressiveModelLoader instance.</returns>
    public static ProgressiveModelLoader LoadProgressive(
        this ModelZoo modelZoo,
        ModelMetadata metadata,
        Device? device = null,
        ProgressiveLoadOptions? options = null)
    {
        if (modelZoo == null)
        {
            throw new ArgumentNullException(nameof(modelZoo));
        }

        return ProgressiveModelLoader.LoadProgressive(metadata, device, options);
    }

    /// <summary>
    /// Loads a model progressively from metadata with a specific loading strategy.
    /// </summary>
    /// <param name="modelZoo">The ModelZoo instance.</param>
    /// <param name="metadata">The model metadata.</param>
    /// <param name="strategy">The loading strategy to use.</param>
    /// <param name="device">The target device (optional).</param>
    /// <returns>A ProgressiveModelLoader instance.</returns>
    public static ProgressiveModelLoader LoadProgressive(
        this ModelZoo modelZoo,
        ModelMetadata metadata,
        LayerLoadingStrategy strategy,
        Device? device = null)
    {
        if (modelZoo == null)
        {
            throw new ArgumentNullException(nameof(modelZoo));
        }

        var options = new ProgressiveLoadOptions(strategy);
        return ProgressiveModelLoader.LoadProgressive(metadata, device, options);
    }
}

/// <summary>
/// Placeholder for ModelZoo class (to be implemented separately).
/// </summary>
public class ModelZoo
{
    // ModelZoo implementation will be in a separate file
}
