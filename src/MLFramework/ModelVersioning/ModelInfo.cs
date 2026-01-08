namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Represents information about a registered model version.
    /// </summary>
    public class ModelInfo
    {
        /// <summary>
        /// Gets or sets the unique identifier for the model.
        /// </summary>
        public string ModelId { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the version tag (e.g., v1.0.0).
        /// </summary>
        public string? VersionTag { get; set; }

        /// <summary>
        /// Gets or sets the file path to the model.
        /// </summary>
        public string ModelPath { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the metadata associated with the model.
        /// </summary>
        public ModelMetadata Metadata { get; set; } = new ModelMetadata();

        /// <summary>
        /// Gets or sets the current lifecycle state of the model.
        /// </summary>
        public LifecycleState State { get; set; } = LifecycleState.Draft;

        /// <summary>
        /// Gets or sets the parent model ID for fine-tuned models.
        /// </summary>
        public string? ParentModelId { get; set; }

        /// <summary>
        /// Gets or sets the timestamp when the model was registered.
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets the timestamp when the model was last updated.
        /// </summary>
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
    }
}
