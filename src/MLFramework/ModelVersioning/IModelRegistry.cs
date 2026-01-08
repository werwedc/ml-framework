namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Interface for registering, tagging, and querying model versions with metadata persistence.
    /// </summary>
    public interface IModelRegistry
    {
        /// <summary>
        /// Registers a model with the specified path and metadata.
        /// </summary>
        /// <param name="modelPath">The file path to the model.</param>
        /// <param name="metadata">The metadata associated with the model.</param>
        /// <returns>The unique model identifier.</returns>
        string RegisterModel(string modelPath, ModelMetadata metadata);

        /// <summary>
        /// Associates a version tag with a model.
        /// </summary>
        /// <param name="modelId">The model identifier.</param>
        /// <param name="versionTag">The version tag (e.g., v1.0.0).</param>
        void TagModel(string modelId, string versionTag);

        /// <summary>
        /// Retrieves model information by version tag.
        /// </summary>
        /// <param name="versionTag">The version tag.</param>
        /// <returns>The model information, or null if not found.</returns>
        ModelInfo? GetModel(string versionTag);

        /// <summary>
        /// Retrieves model information by model ID.
        /// </summary>
        /// <param name="modelId">The model identifier.</param>
        /// <returns>The model information, or null if not found.</returns>
        ModelInfo? GetModelById(string modelId);

        /// <summary>
        /// Lists all registered models.
        /// </summary>
        /// <returns>A collection of model information.</returns>
        IEnumerable<ModelInfo> ListModels();

        /// <summary>
        /// Lists registered models filtered by state.
        /// </summary>
        /// <param name="state">The lifecycle state to filter by.</param>
        /// <returns>A collection of model information matching the state.</returns>
        IEnumerable<ModelInfo> ListModels(LifecycleState state);

        /// <summary>
        /// Updates the lifecycle state of a model.
        /// </summary>
        /// <param name="modelId">The model identifier.</param>
        /// <param name="newState">The new lifecycle state.</param>
        void UpdateModelState(string modelId, LifecycleState newState);

        /// <summary>
        /// Sets the parent model ID for tracking fine-tuning lineage.
        /// </summary>
        /// <param name="modelId">The model identifier.</param>
        /// <param name="parentModelId">The parent model identifier.</param>
        void SetParentModel(string modelId, string parentModelId);
    }
}
