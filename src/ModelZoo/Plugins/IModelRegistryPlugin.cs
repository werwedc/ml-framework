using System.Threading.Tasks;

namespace MLFramework.ModelZoo.Plugins
{
    /// <summary>
    /// Base interface for model registry plugins.
    /// Plugins implement this interface to provide custom model registries to the Model Zoo.
    /// </summary>
    public interface IModelRegistryPlugin
    {
        /// <summary>
        /// Gets the registry identifier.
        /// </summary>
        string RegistryName { get; }

        /// <summary>
        /// Gets the plugin priority. Higher values are checked first when finding a plugin.
        /// </summary>
        int Priority { get; }

        /// <summary>
        /// Gets the metadata for a specific model.
        /// </summary>
        /// <param name="modelName">Name of the model.</param>
        /// <param name="version">Optional version string. If null, gets the latest version.</param>
        /// <returns>Model metadata.</returns>
        Task<ModelVersioning.ModelMetadata> GetModelMetadataAsync(string modelName, string version = null);

        /// <summary>
        /// Downloads the model file(s).
        /// </summary>
        /// <param name="metadata">Model metadata containing download information.</param>
        /// <param name="progress">Optional progress reporter.</param>
        /// <returns>Stream containing the model data.</returns>
        Task<System.IO.Stream> DownloadModelAsync(ModelVersioning.ModelMetadata metadata, IProgress<double> progress = null);

        /// <summary>
        /// Checks if a model exists in the registry.
        /// </summary>
        /// <param name="modelName">Name of the model.</param>
        /// <param name="version">Optional version string. If null, checks for any version.</param>
        /// <returns>True if the model exists, false otherwise.</returns>
        Task<bool> ModelExistsAsync(string modelName, string version = null);

        /// <summary>
        /// Lists all models available in this registry.
        /// </summary>
        /// <returns>Array of model names.</returns>
        Task<string[]> ListModelsAsync();

        /// <summary>
        /// Determines if this registry can handle the specified model name.
        /// </summary>
        /// <param name="modelName">Name of the model to check.</param>
        /// <returns>True if this registry can handle the model, false otherwise.</returns>
        bool CanHandle(string modelName);
    }
}
