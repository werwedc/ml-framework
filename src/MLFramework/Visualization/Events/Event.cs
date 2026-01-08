namespace MachineLearning.Visualization.Events;

/// <summary>
/// Base class for all visualization events
/// </summary>
public abstract class Event
{
    /// <summary>
    /// Timestamp when the event was created (UTC)
    /// </summary>
    public DateTime Timestamp { get; }

    /// <summary>
    /// Unique identifier for this event
    /// </summary>
    public Guid EventId { get; }

    /// <summary>
    /// Event type name for serialization
    /// </summary>
    public string EventType => GetType().Name;

    protected Event()
    {
        Timestamp = DateTime.UtcNow;
        EventId = Guid.NewGuid();
    }

    /// <summary>
    /// Serializes the event to bytes for storage
    /// </summary>
    public abstract byte[] Serialize();

    /// <summary>
    /// Deserializes bytes back to an event
    /// </summary>
    public abstract void Deserialize(byte[] data);
}
