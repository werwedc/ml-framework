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
    /// <param name="operation">The operation to enqueue</param>
    void Enqueue(Action operation);

    /// <summary>
    /// Record an event at the current point in this stream
    /// </summary>
    /// <returns>A new event representing the current stream position</returns>
    IEvent RecordEvent();

    /// <summary>
    /// Wait for an event to complete before continuing
    /// </summary>
    /// <param name="event">The event to wait for</param>
    void Wait(IEvent @event);

    /// <summary>
    /// Synchronize this stream (block until all operations complete)
    /// </summary>
    void Synchronize();
}
