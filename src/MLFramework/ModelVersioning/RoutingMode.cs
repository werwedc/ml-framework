using System.Text.Json.Serialization;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Defines the mode for routing requests to different model versions
    /// </summary>
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public enum RoutingMode
    {
        /// <summary>
        /// Route based on percentage split across versions
        /// </summary>
        Percentage,

        /// <summary>
        /// Route to multiple versions, return from primary for comparison
        /// </summary>
        Shadow,

        /// <summary>
        /// Route based on user ID, segment, or region
        /// </summary>
        Deterministic,

        /// <summary>
        /// Route based on time-based schedule
        /// </summary>
        TimeBased
    }
}
