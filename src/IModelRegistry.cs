using System.Collections.Generic;

namespace MLFramework.ModelRegistry
{
    /// <summary>
    /// Interface for a centralized registry of model versions with metadata support.
    /// All operations are thread-safe for concurrent access.
    /// </summary>
    public interface IModelRegistry
    {
        /// <summary>
        /// Registers a model with its metadata.
        /// </summary>
        /// <param name="name">The model name.</param>
        /// <param name="version">The semantic version string.</param>
        /// <param name="metadata">The model metadata.</param>
        /// <exception cref="InvalidOperationException">Thrown if the model version is already registered.</exception>
        void RegisterModel(string name, string version, ModelMetadata metadata);

        /// <summary>
        /// Unregisters a model version.
        /// </summary>
        /// <param name="name">The model name.</param>
        /// <param name="version">The semantic version string.</param>
        /// <exception cref="KeyNotFoundException">Thrown if the model version is not found.</exception>
        void UnregisterModel(string name, string version);

        /// <summary>
        /// Checks if a specific version of a model exists.
        /// </summary>
        /// <param name="name">The model name.</param>
        /// <param name="version">The semantic version string.</param>
        /// <returns>True if the version exists, false otherwise.</returns>
        bool HasVersion(string name, string version);

        /// <summary>
        /// Gets all registered versions for a model.
        /// </summary>
        /// <param name="name">The model name.</param>
        /// <returns>A collection of version strings.</returns>
        IEnumerable<string> GetVersions(string name);

        /// <summary>
        /// Gets the metadata for a specific model version.
        /// </summary>
        /// <param name="name">The model name.</param>
        /// <param name="version">The semantic version string.</param>
        /// <returns>The model metadata.</returns>
        /// <exception cref="KeyNotFoundException">Thrown if the model version is not found.</exception>
        ModelMetadata GetMetadata(string name, string version);

        /// <summary>
        /// Gets all registered model names.
        /// </summary>
        /// <returns>A collection of model names.</returns>
        IEnumerable<string> GetAllModelNames();
    }
}
