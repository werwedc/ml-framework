namespace MLFramework.HAL;

/// <summary>
/// Represents a command stream for async operations
/// </summary>
public interface IStream : IDisposable
{
    /// <summary>
    /// Device this stream belongs to
    /// </summary>
    IDevice Device { get; }

    /// <summary>
    /// Enqueue an operation to be executed on this stream
    /// </summary>
    void Enqueue(Action operation);

    /// <summary>
    /// Record an event at the current point in this stream
    /// </summary>
    IEvent RecordEvent();

    /// <summary>
    /// Wait for an event to complete before continuing
    /// </summary>
    void Wait(IEvent @event);

    /// <summary>
    /// Synchronize this stream (block until all operations complete)
    /// </summary>
    void Synchronize();
}
