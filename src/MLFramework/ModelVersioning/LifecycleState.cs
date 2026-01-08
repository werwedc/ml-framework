using System.Text.Json.Serialization;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Lifecycle states for a model version
    /// </summary>
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public enum LifecycleState
    {
        /// <summary>
        /// Model is in draft state, still being developed
        /// </summary>
        Draft,

        /// <summary>
        /// Model is ready for staging/testing
        /// </summary>
        Staging,

        /// <summary>
        /// Model is in production use
        /// </summary>
        Production,

        /// <summary>
        /// Model is archived and no longer in use
        /// </summary>
        Archived
    }
}
