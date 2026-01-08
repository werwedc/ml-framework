using System.Text.Json.Serialization;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Defines a routing policy for directing traffic to different model versions
    /// </summary>
    public class RoutingPolicy
    {
        /// <summary>
        /// The unique identifier of the model this policy applies to
        /// </summary>
        [JsonPropertyName("modelId")]
        public string? ModelId { get; set; }

        /// <summary>
        /// List of routing rules that define how traffic should be directed
        /// </summary>
        [JsonPropertyName("rules")]
        public List<RoutingRule>? Rules { get; set; }

        /// <summary>
        /// The routing mode to use for this policy
        /// </summary>
        [JsonPropertyName("mode")]
        public RoutingMode Mode { get; set; }

        /// <summary>
        /// The date and time when this policy becomes effective
        /// </summary>
        [JsonPropertyName("effectiveDate")]
        public DateTime EffectiveDate { get; set; }

        /// <summary>
        /// Creates a new RoutingPolicy
        /// </summary>
        public RoutingPolicy()
        {
            Rules = new List<RoutingRule>();
            EffectiveDate = DateTime.UtcNow;
        }

        /// <summary>
        /// Creates a string representation of the routing policy
        /// </summary>
        public override string ToString()
        {
            return $"RoutingPolicy(ModelId: {ModelId}, Mode: {Mode}, Rules: {Rules?.Count ?? 0}, EffectiveDate: {EffectiveDate})";
        }
    }
}
