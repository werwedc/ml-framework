using System.Text.Json;

namespace MachineLearning.Visualization.Events;

/// <summary>
/// Event representing a scalar metric value (e.g., loss, accuracy, learning rate)
/// </summary>
public class ScalarMetricEvent : Event
{
    /// <summary>
    /// Name of the metric
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Value of the metric
    /// </summary>
    public float Value { get; }

    /// <summary>
    /// Training step at which this metric was recorded
    /// </summary>
    public long Step { get; }

    /// <summary>
    /// Additional metadata for this metric
    /// </summary>
    public Dictionary<string, string> Metadata { get; }

    /// <summary>
    /// Creates a new scalar metric event
    /// </summary>
    /// <param name="name">Name of the metric</param>
    /// <param name="value">Value of the metric</param>
    /// <param name="step">Training step</param>
    /// <param name="metadata">Optional metadata</param>
    public ScalarMetricEvent(string name, float value, long step = -1, Dictionary<string, string>? metadata = null)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Value = value;
        Step = step;
        Metadata = metadata ?? new Dictionary<string, string>();
    }

    /// <summary>
    /// Serializes the event to bytes
    /// </summary>
    public override byte[] Serialize()
    {
        var data = new Dictionary<string, object>
        {
            { "type", MachineLearning.Visualization.Events.EventType.ScalarMetric.ToString() },
            { "name", Name },
            { "value", Value },
            { "step", Step },
            { "timestamp", Timestamp.ToString("o") },
            { "eventId", EventId.ToString() },
            { "metadata", Metadata }
        };

        return JsonSerializer.SerializeToUtf8Bytes(data);
    }

    /// <summary>
    /// Deserializes bytes back to an event
    /// </summary>
    public override void Deserialize(byte[] data)
    {
        // Note: In a real implementation, we would update the properties from the data
        // This is a placeholder that shows the concept
        var json = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(data);
        // Implementation would parse and update properties
    }
}
