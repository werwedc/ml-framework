using System.Text.Json;

namespace MachineLearning.Visualization.Events;

/// <summary>
/// Event representing a memory allocation (e.g., tensor allocation)
/// </summary>
public class MemoryAllocationEvent : Event
{
    /// <summary>
    /// Name of the allocated tensor or buffer
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Size of the allocation in bytes
    /// </summary>
    public long SizeBytes { get; }

    /// <summary>
    /// Training step at which this allocation occurred
    /// </summary>
    public long Step { get; }

    /// <summary>
    /// Memory location (CPU, GPU, etc.)
    /// </summary>
    public string Location { get; }

    /// <summary>
    /// Additional metadata for this allocation
    /// </summary>
    public Dictionary<string, string> Metadata { get; }

    /// <summary>
    /// Creates a new memory allocation event
    /// </summary>
    /// <param name="name">Name of the allocation</param>
    /// <param name="sizeBytes">Size in bytes</param>
    /// <param name="step">Training step</param>
    /// <param name="location">Memory location</param>
    /// <param name="metadata">Optional metadata</param>
    public MemoryAllocationEvent(string name, long sizeBytes, long step = -1, string location = "CPU", Dictionary<string, string>? metadata = null)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        SizeBytes = sizeBytes;
        Step = step;
        Location = location ?? "CPU";
        Metadata = metadata ?? new Dictionary<string, string>();
    }

    /// <summary>
    /// Serializes the event to bytes
    /// </summary>
    public override byte[] Serialize()
    {
        var data = new Dictionary<string, object>
        {
            { "type", MachineLearning.Visualization.Events.EventType.MemoryAllocation.ToString() },
            { "name", Name },
            { "sizeBytes", SizeBytes },
            { "step", Step },
            { "location", Location },
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
