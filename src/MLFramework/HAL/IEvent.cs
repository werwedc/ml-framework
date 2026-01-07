namespace MLFramework.HAL;

/// <summary>
/// Represents a synchronization point in a stream
/// </summary>
public interface IEvent : IDisposable
{
    /// <summary>
    /// Stream that recorded this event
    /// </summary>
    IStream Stream { get; }

    /// <summary>
    /// Check if the event has completed (non-blocking)
    /// </summary>
    bool IsCompleted { get; }

    /// <summary>
    /// Block until this event completes
    /// </summary>
    void Synchronize();
}
