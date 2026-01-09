using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.ComponentModel.DataAnnotations;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Represents metadata for a pre-trained model in the Model Zoo.
    /// </summary>
    public class ModelMetadata
    {
        /// <summary>
        /// Gets or sets the model name/identifier.
        /// </summary>
        [Required(ErrorMessage = "Model name is required")]
        [JsonPropertyName("name")]
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the semantic version (e.g., "1.0.0").
        /// </summary>
        [Required(ErrorMessage = "Model version is required")]
        [JsonPropertyName("version")]
        public string Version { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the architecture type (e.g., "ResNet", "BERT").
        /// </summary>
        [Required(ErrorMessage = "Architecture is required")]
        [JsonPropertyName("architecture")]
        public string Architecture { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the task type for the model.
        /// </summary>
        [JsonPropertyName("task")]
        public TaskType Task { get; set; }

        /// <summary>
        /// Gets or sets the available variants (e.g., ["resnet18", "resnet50"]).
        /// </summary>
        [JsonPropertyName("variants")]
        public string[] Variants { get; set; } = Array.Empty<string>();

        /// <summary>
        /// Gets or sets the dataset used for pre-training.
        /// </summary>
        [Required(ErrorMessage = "Pre-training dataset is required")]
        [JsonPropertyName("pretrained_on")]
        public string PretrainedOn { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets performance metrics (e.g., accuracy, F1-score, top1, top5).
        /// </summary>
        [JsonPropertyName("performance_metrics")]
        public Dictionary<string, double> PerformanceMetrics { get; set; } = new Dictionary<string, double>();

        /// <summary>
        /// Gets or sets the expected input dimensions (e.g., [3, 224, 224]).
        /// </summary>
        [Required(ErrorMessage = "Input shape is required")]
        [JsonPropertyName("input_shape")]
        public int[] InputShape { get; set; } = Array.Empty<int>();

        /// <summary>
        /// Gets or sets the expected output dimensions.
        /// </summary>
        [JsonPropertyName("output_shape")]
        public int[] OutputShape { get; set; } = Array.Empty<int>();

        /// <summary>
        /// Gets or sets the number of model parameters.
        /// </summary>
        [JsonPropertyName("num_parameters")]
        public long NumParameters { get; set; }

        /// <summary>
        /// Gets or sets the size of the model file in bytes.
        /// </summary>
        [JsonPropertyName("file_size_bytes")]
        public long FileSizeBytes { get; set; }

        /// <summary>
        /// Gets or sets the license type (e.g., "MIT", "Apache-2.0").
        /// </summary>
        [Required(ErrorMessage = "License is required")]
        [JsonPropertyName("license")]
        public string License { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the link to the research paper.
        /// </summary>
        [JsonPropertyName("paper_url")]
        public string? PaperUrl { get; set; }

        /// <summary>
        /// Gets or sets the link to the source code.
        /// </summary>
        [JsonPropertyName("source_code_url")]
        public string? SourceCodeUrl { get; set; }

        /// <summary>
        /// Gets or sets the SHA256 hash for file integrity validation.
        /// </summary>
        [Required(ErrorMessage = "SHA256 checksum is required")]
        [JsonPropertyName("sha256_checksum")]
        public string Sha256Checksum { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the primary download URL.
        /// </summary>
        [Required(ErrorMessage = "Download URL is required")]
        [JsonPropertyName("download_url")]
        public string DownloadUrl { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets fallback download URLs.
        /// </summary>
        [JsonPropertyName("mirror_urls")]
        public string[] MirrorUrls { get; set; } = Array.Empty<string>();

        /// <summary>
        /// Serializes the metadata to JSON.
        /// </summary>
        public string ToJson()
        {
            return JsonSerializer.Serialize(this, new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
            });
        }

        /// <summary>
        /// Deserializes metadata from JSON.
        /// </summary>
        public static ModelMetadata FromJson(string json)
        {
            return JsonSerializer.Deserialize<ModelMetadata>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            }) ?? throw new ArgumentException("Invalid JSON format for ModelMetadata");
        }

        /// <summary>
        /// Serializes the metadata to a JSON file.
        /// </summary>
        public void SaveToJsonFile(string filePath)
        {
            string json = ToJson();
            File.WriteAllText(filePath, json);
        }

        /// <summary>
        /// Loads metadata from a JSON file.
        /// </summary>
        public static ModelMetadata LoadFromJsonFile(string filePath)
        {
            string json = File.ReadAllText(filePath);
            return FromJson(json);
        }
    }
}
