using System.Text.Json.Serialization;

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
        [JsonPropertyName("version")]
        public string Version { get; set; } = string.Empty;

        /// <summary>
        /// Date when the model was trained.
        /// </summary>
        [JsonPropertyName("trainingDate")]
        public DateTime TrainingDate { get; set; }

        /// <summary>
        /// Hyperparameters used during training.
        /// </summary>
        [JsonPropertyName("hyperparameters")]
        public Dictionary<string, object> Hyperparameters { get; set; } = new();

        /// <summary>
        /// Performance metrics for model.
        /// </summary>
        [JsonPropertyName("performanceMetrics")]
        public Dictionary<string, float> PerformanceMetrics { get; set; } = new();

        /// <summary>
        /// Path or URI to the model artifact.
        /// </summary>
        [JsonPropertyName("artifactPath")]
        public string ArtifactPath { get; set; } = string.Empty;
    }
}
}
