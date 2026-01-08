using System.Text.Json.Serialization;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Information about a registered model version
    /// </summary>
    public class ModelInfo
    {
        /// <summary>
        /// Unique identifier for the model
        /// </summary>
        [JsonPropertyName("modelId")]
        public string? ModelId { get; set; }

        /// <summary>
        /// Human-readable name of the model
        /// </summary>
        [JsonPropertyName("name")]
        public string? Name { get; set; }

        /// <summary>
        /// Version tag (e.g., v1.2.3)
        /// </summary>
        [JsonPropertyName("versionTag")]
        public string? VersionTag { get; set; }

        /// <summary>
        /// Model metadata
        /// </summary>
        [JsonPropertyName("metadata")]
        public ModelMetadata? Metadata { get; set; }

        /// <summary>
        /// Current lifecycle state
        /// </summary>
        [JsonPropertyName("state")]
        public LifecycleState State { get; set; }

        /// <summary>
        /// Parent model ID (for fine-tuning lineage)
        /// </summary>
        [JsonPropertyName("parentModelId")]
        public string? ParentModelId { get; set; }

        /// <summary>
        /// Creates a string representation of the model info
        /// </summary>
        public override string ToString()
        {
            return $"ModelInfo(Id: {ModelId}, Name: {Name}, Version: {VersionTag ?? "N/A"}, State: {State})";
        }
    }
}
