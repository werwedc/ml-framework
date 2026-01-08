using System.Text.Json;

namespace MachineLearning.Visualization.Events;

/// <summary>
/// Event representing the start of a profiling operation
/// </summary>
public class ProfilingStartEvent : Event
{
    /// <summary>
    /// Name of the operation being profiled
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Training step at which profiling started
    /// </summary>
    public long Step { get; }

    /// <summary>
    /// Thread ID for this operation
    /// </summary>
    public int ThreadId { get; }

    /// <summary>
    /// Additional metadata for this profiling operation
    /// </summary>
    public Dictionary<string, string> Metadata { get; }

    /// <summary>
    /// Creates a new profiling start event
    /// </summary>
    /// <param name="name">Name of the operation</param>
    /// <param name="step">Training step</param>
    /// <param name="metadata">Optional metadata</param>
    public ProfilingStartEvent(string name, long step = -1, Dictionary<string, string>? metadata = null)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Step = step;
        ThreadId = Environment.CurrentManagedThreadId;
        Metadata = metadata ?? new Dictionary<string, string>();
    }

    /// <summary>
    /// Serializes the event to bytes
    /// </summary>
    public override byte[] Serialize()
    {
        var data = new Dictionary<string, object>
        {
            { "type", MachineLearning.Visualization.Events.EventType.ProfilingStart.ToString() },
            { "name", Name },
            { "step", Step },
            { "threadId", ThreadId },
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

/// <summary>
/// Event representing the end of a profiling operation
/// </summary>
public class ProfilingEndEvent : Event
{
    /// <summary>
    /// Name of the operation that was profiled
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Training step at which profiling ended
    /// </summary>
    public long Step { get; }

    /// <summary>
    /// Thread ID for this operation
    /// </summary>
    public int ThreadId { get; }

    /// <summary>
    /// Duration of the operation in nanoseconds
    /// </summary>
    public long DurationNanoseconds { get; }

    /// <summary>
    /// Additional metadata for this profiling operation
    /// </summary>
    public Dictionary<string, string> Metadata { get; }

    /// <summary>
    /// Creates a new profiling end event
    /// </summary>
    /// <param name="name">Name of the operation</param>
    /// <param name="step">Training step</param>
    /// <param name="durationNanoseconds">Duration of the operation</param>
    /// <param name="metadata">Optional metadata</param>
    public ProfilingEndEvent(string name, long step = -1, long durationNanoseconds = 0, Dictionary<string, string>? metadata = null)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Step = step;
        ThreadId = Environment.CurrentManagedThreadId;
        DurationNanoseconds = durationNanoseconds;
        Metadata = metadata ?? new Dictionary<string, string>();
    }

    /// <summary>
    /// Serializes the event to bytes
    /// </summary>
    public override byte[] Serialize()
    {
        var data = new Dictionary<string, object>
        {
            { "type", MachineLearning.Visualization.Events.EventType.ProfilingEnd.ToString() },
            { "name", Name },
            { "step", Step },
            { "threadId", ThreadId },
            { "durationNanoseconds", DurationNanoseconds },
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
