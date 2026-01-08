using System.Text.Json.Serialization;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Context information for a routing request
    /// </summary>
    public class RequestContext
    {
        /// <summary>
        /// Unique identifier for the user making the request
        /// </summary>
        [JsonPropertyName("userId")]
        public string? UserId { get; set; }

        /// <summary>
        /// User segment for deterministic routing
        /// </summary>
        [JsonPropertyName("segment")]
        public string? Segment { get; set; }

        /// <summary>
        /// Geographic region for routing decisions
        /// </summary>
        [JsonPropertyName("region")]
        public string? Region { get; set; }

        /// <summary>
        /// The timestamp when the request was made
        /// </summary>
        [JsonPropertyName("requestTime")]
        public DateTime RequestTime { get; set; }

        /// <summary>
        /// Additional metadata for the request
        /// </summary>
        [JsonPropertyName("metadata")]
        public Dictionary<string, string>? Metadata { get; set; }

        /// <summary>
        /// Creates a new RequestContext
        /// </summary>
        public RequestContext()
        {
            RequestTime = DateTime.UtcNow;
            Metadata = new Dictionary<string, string>();
        }

        /// <summary>
        /// Creates a string representation of the request context
        /// </summary>
        public override string ToString()
        {
            return $"RequestContext(UserId: {UserId}, Segment: {Segment}, Region: {Region}, RequestTime: {RequestTime})";
        }
    }
}
