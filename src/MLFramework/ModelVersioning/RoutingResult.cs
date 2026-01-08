using System.Text.Json.Serialization;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Result of routing a request to a specific model version
    /// </summary>
    public class RoutingResult
    {
        /// <summary>
        /// The version that should handle this request
        /// </summary>
        [JsonPropertyName("version")]
        public string? Version { get; set; }

        /// <summary>
        /// Indicates if this is a shadow routing result
        /// </summary>
        [JsonPropertyName("isShadow")]
        public bool IsShadow { get; set; }

        /// <summary>
        /// List of shadow versions for comparison (in Shadow mode)
        /// </summary>
        [JsonPropertyName("shadowVersions")]
        public List<string>? ShadowVersions { get; set; }

        /// <summary>
        /// Description of which rule matched
        /// </summary>
        [JsonPropertyName("ruleMatched")]
        public string? RuleMatched { get; set; }

        /// <summary>
        /// Creates a new RoutingResult
        /// </summary>
        public RoutingResult()
        {
            ShadowVersions = new List<string>();
        }

        /// <summary>
        /// Creates a string representation of the routing result
        /// </summary>
        public override string ToString()
        {
            return $"RoutingResult(Version: {Version}, IsShadow: {IsShadow}, ShadowVersions: {ShadowVersions?.Count ?? 0}, Rule: {RuleMatched})";
        }
    }
}
