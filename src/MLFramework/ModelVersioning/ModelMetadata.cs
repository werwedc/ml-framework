using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Metadata associated with a machine learning model
    /// </summary>
    public class ModelMetadata
    {
        /// <summary>
        /// Timestamp when the model was created
        /// </summary>
        [JsonPropertyName("creationTimestamp")]
        public DateTime CreationTimestamp { get; set; }

        /// <summary>
        /// Training parameters used to create the model
        /// </summary>
        [JsonPropertyName("trainingParameters")]
        public Dictionary<string, object>? TrainingParameters { get; set; }

        /// <summary>
        /// Performance metrics of the model
        /// </summary>
        [JsonPropertyName("performance")]
        public PerformanceMetrics? Performance { get; set; }

        /// <summary>
        /// Version of the dataset used for training
        /// </summary>
        [JsonPropertyName("datasetVersion")]
        public string? DatasetVersion { get; set; }

        /// <summary>
        /// Hash of the model architecture
        /// </summary>
        [JsonPropertyName("architectureHash")]
        public string? ArchitectureHash { get; set; }

        /// <summary>
        /// Custom metadata key-value pairs
        /// </summary>
        [JsonPropertyName("customMetadata")]
        public Dictionary<string, string>? CustomMetadata { get; set; }

        /// <summary>
        /// Creates a string representation of the metadata
        /// </summary>
        public override string ToString()
        {
            return $"ModelMetadata(Created: {CreationTimestamp}, Dataset: {DatasetVersion ?? "N/A"}, ArchitectureHash: {ArchitectureHash ?? "N/A"})";
        }
    }
}
