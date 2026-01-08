namespace MLFramework.Checkpointing.Profiling;

/// <summary>
/// Represents a checkpoint event
/// </summary>
public class CheckpointEvent
{
    /// <summary>
    /// Layer ID
    /// </summary>
    public string LayerId { get; set; } = string.Empty;

    /// <summary>
    /// Type of event
    /// </summary>
    public CheckpointEventType EventType { get; set; }

    /// <summary>
    /// Timestamp of the event
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Duration in milliseconds
    /// </summary>
    public long DurationMs { get; set; }

    /// <summary>
    /// Memory affected in bytes
    /// </summary>
    public long MemoryBytes { get; set; }
}
