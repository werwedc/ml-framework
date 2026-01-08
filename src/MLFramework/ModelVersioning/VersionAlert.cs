using System;
using System.Collections.Generic;

namespace MLFramework.ModelVersioning
{
    /// <summary>
    /// Represents an alert for a model version based on metrics or anomalies.
    /// </summary>
    public class VersionAlert
    {
        /// <summary>
        /// Unique identifier for the alert.
        /// </summary>
        public string AlertId { get; set; }

        /// <summary>
        /// Model ID that triggered the alert.
        /// </summary>
        public string ModelId { get; set; }

        /// <summary>
        /// Version tag that triggered the alert.
        /// </summary>
        public string Version { get; set; }

        /// <summary>
        /// Type of alert.
        /// </summary>
        public AlertType Type { get; set; }

        /// <summary>
        /// Human-readable alert message.
        /// </summary>
        public string Message { get; set; }

        /// <summary>
        /// Timestamp when the alert was triggered.
        /// </summary>
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// Additional context about the alert (thresholds, actual values, etc.).
        /// </summary>
        public Dictionary<string, object> Context { get; set; }

        /// <summary>
        /// Severity level of the alert.
        /// </summary>
        public AlertSeverity Severity { get; set; }

        /// <summary>
        /// Creates a new version alert.
        /// </summary>
        public static VersionAlert Create(string modelId, string version, AlertType type, string message, AlertSeverity severity)
        {
            return new VersionAlert
            {
                AlertId = Guid.NewGuid().ToString(),
                ModelId = modelId,
                Version = version,
                Type = type,
                Message = message,
                Timestamp = DateTime.UtcNow,
                Context = new Dictionary<string, object>(),
                Severity = severity
            };
        }

        /// <summary>
        /// Creates a new version alert with context.
        /// </summary>
        public static VersionAlert Create(string modelId, string version, AlertType type, string message, AlertSeverity severity, Dictionary<string, object> context)
        {
            return new VersionAlert
            {
                AlertId = Guid.NewGuid().ToString(),
                ModelId = modelId,
                Version = version,
                Type = type,
                Message = message,
                Timestamp = DateTime.UtcNow,
                Context = context ?? new Dictionary<string, object>(),
                Severity = severity
            };
        }

        /// <summary>
        /// Validates that the alert has all required fields.
        /// </summary>
        public bool IsValid()
        {
            return !string.IsNullOrEmpty(AlertId) &&
                   !string.IsNullOrEmpty(ModelId) &&
                   !string.IsNullOrEmpty(Version) &&
                   !string.IsNullOrEmpty(Message) &&
                   Timestamp != default;
        }

        /// <summary>
        /// Creates a copy of this alert.
        /// </summary>
        public VersionAlert Clone()
        {
            return new VersionAlert
            {
                AlertId = AlertId,
                ModelId = ModelId,
                Version = Version,
                Type = Type,
                Message = Message,
                Timestamp = Timestamp,
                Context = new Dictionary<string, object>(Context),
                Severity = Severity
            };
        }

        /// <summary>
        /// Formats the alert as a human-readable string.
        /// </summary>
        public override string ToString()
        {
            return $"[{Severity}] {Type}: {Message} (Model: {ModelId}, Version: {Version}, Time: {Timestamp:yyyy-MM-dd HH:mm:ss})";
        }
    }
}
