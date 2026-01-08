using System.Text.Json.Serialization;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Defines a routing rule for directing traffic to specific model versions
    /// </summary>
    public class RoutingRule
    {
        /// <summary>
        /// The target version for this rule
        /// </summary>
        [JsonPropertyName("version")]
        public string? Version { get; set; }

        /// <summary>
        /// Percentage of traffic to route (0-100) for Percentage mode
        /// </summary>
        [JsonPropertyName("percentage")]
        public double Percentage { get; set; }

        /// <summary>
        /// Regex pattern for user ID matching for Deterministic mode
        /// </summary>
        [JsonPropertyName("userIdPattern")]
        public string? UserIdPattern { get; set; }

        /// <summary>
        /// User segment for Deterministic mode
        /// </summary>
        [JsonPropertyName("segment")]
        public string? Segment { get; set; }

        /// <summary>
        /// Geographic region for Deterministic mode
        /// </summary>
        [JsonPropertyName("region")]
        public string? Region { get; set; }

        /// <summary>
        /// Time range for TimeBased mode
        /// </summary>
        [JsonPropertyName("timeRange")]
        public TimeRange? TimeRange { get; set; }

        /// <summary>
        /// Indicates if this is the primary version for Shadow mode
        /// </summary>
        [JsonPropertyName("isPrimary")]
        public bool IsPrimary { get; set; }

        /// <summary>
        /// Creates a string representation of the routing rule
        /// </summary>
        public override string ToString()
        {
            return $"RoutingRule(Version: {Version}, Percentage: {Percentage}, Segment: {Segment}, Region: {Region}, IsPrimary: {IsPrimary})";
        }
    }
}
