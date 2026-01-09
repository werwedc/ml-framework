using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MLFramework.ModelZoo
{
    /// <summary>
    /// Stores cache tracking information for a cached model.
    /// </summary>
    public class CacheMetadata
    {
        /// <summary>
        /// Gets or sets the last time the model was accessed.
        /// </summary>
        [JsonPropertyName("lastAccessed")]
        public DateTime LastAccessed { get; set; }

        /// <summary>
        /// Gets or sets when the model was downloaded.
        /// </summary>
        [JsonPropertyName("downloadDate")]
        public DateTime DownloadDate { get; set; }

        /// <summary>
        /// Gets or sets the size of the cached file in bytes.
        /// </summary>
        [JsonPropertyName("fileSize")]
        public long FileSize { get; set; }

        /// <summary>
        /// Gets or sets the number of times the model was loaded.
        /// </summary>
        [JsonPropertyName("accessCount")]
        public int AccessCount { get; set; }

        /// <summary>
        /// Gets or sets the model version.
        /// </summary>
        [JsonPropertyName("version")]
        public string Version { get; set; }

        /// <summary>
        /// Gets or sets the model name.
        /// </summary>
        [JsonPropertyName("modelName")]
        public string ModelName { get; set; }

        /// <summary>
        /// Gets or sets the file checksum (e.g., SHA-256) for integrity verification.
        /// </summary>
        [JsonPropertyName("checksum")]
        public string? Checksum { get; set; }

        /// <summary>
        /// Initializes a new instance of the CacheMetadata class.
        /// </summary>
        public CacheMetadata()
        {
            LastAccessed = DateTime.UtcNow;
            DownloadDate = DateTime.UtcNow;
            FileSize = 0;
            AccessCount = 0;
            Version = string.Empty;
            ModelName = string.Empty;
            Checksum = null;
        }

        /// <summary>
        /// Serializes the metadata to JSON.
        /// </summary>
        /// <returns>JSON string representation of the metadata.</returns>
        public string ToJson()
        {
            var options = new JsonSerializerOptions
            {
                WriteIndented = true
            };
            return JsonSerializer.Serialize(this, options);
        }

        /// <summary>
        /// Deserializes metadata from JSON.
        /// </summary>
        /// <param name="json">JSON string to deserialize.</param>
        /// <returns>CacheMetadata instance.</returns>
        public static CacheMetadata FromJson(string json)
        {
            return JsonSerializer.Deserialize<CacheMetadata>(json) 
                   ?? throw new ArgumentException("Invalid metadata JSON");
        }

        /// <summary>
        /// Updates the last accessed time and increments the access count.
        /// </summary>
        public void RecordAccess()
        {
            LastAccessed = DateTime.UtcNow;
            AccessCount++;
        }
    }
}
