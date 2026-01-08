namespace MLFramework.ModelRegistry
{
    /// <summary>
    /// Metadata associated with a registered model version.
    /// </summary>
    public class ModelMetadata
    {
        /// <summary>
        /// Semantic version string (MAJOR.MINOR.PATCH format).
        /// </summary>
        public string Version { get; set; } = string.Empty;

        /// <summary>
        /// Date when the model was trained.
        /// </summary>
        public DateTime TrainingDate { get; set; }

        /// <summary>
        /// Hyperparameters used during training.
        /// </summary>
        public Dictionary<string, object> Hyperparameters { get; set; } = new();

        /// <summary>
        /// Performance metrics for the model.
        /// </summary>
        public Dictionary<string, float> PerformanceMetrics { get; set; } = new();

        /// <summary>
        /// Path or URI to the model artifact.
        /// </summary>
        public string ArtifactPath { get; set; } = string.Empty;
    }
}
