namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Interface for managing model version loading, unloading, and lifecycle.
    /// </summary>
    public interface IModelVersionManager
    {
        /// <summary>
        /// Loads a specific version of a model from the given path.
        /// </summary>
        /// <param name="modelId">The identifier of the model.</param>
        /// <param name="version">The version of the model to load.</param>
        /// <param name="modelPath">The path to the model file.</param>
        void LoadVersion(string modelId, string version, string modelPath);

        /// <summary>
        /// Unloads a specific version of a model.
        /// </summary>
        /// <param name="modelId">The identifier of the model.</param>
        /// <param name="version">The version of the model to unload.</param>
        void UnloadVersion(string modelId, string version);

        /// <summary>
        /// Checks if a specific version of a model is currently loaded.
        /// </summary>
        /// <param name="modelId">The identifier of the model.</param>
        /// <param name="version">The version to check.</param>
        /// <returns>True if the version is loaded, otherwise false.</returns>
        bool IsVersionLoaded(string modelId, string version);

        /// <summary>
        /// Gets all loaded versions for a specific model.
        /// </summary>
        /// <param name="modelId">The identifier of the model.</param>
        /// <returns>An enumerable of loaded version strings.</returns>
        IEnumerable<string> GetLoadedVersions(string modelId);

        /// <summary>
        /// Warms up a specific version of a model using the provided warmup data.
        /// </summary>
        /// <param name="modelId">The identifier of the model.</param>
        /// <param name="version">The version to warm up.</param>
        /// <param name="warmupData">The warmup data to use.</param>
        void WarmUpVersion(string modelId, string version, IEnumerable<object> warmupData);

        /// <summary>
        /// Gets load information for a specific version of a model.
        /// </summary>
        /// <param name="modelId">The identifier of the model.</param>
        /// <param name="version">The version to get information for.</param>
        /// <returns>The load information for the version.</returns>
        VersionLoadInfo GetLoadInfo(string modelId, string version);
    }
}
