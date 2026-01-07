namespace MLFramework.HAL;

/// <summary>
/// CPU stream that executes operations synchronously
/// </summary>
public class CpuStream : IStream
{
    private readonly Queue<System.Action> _pendingOperations = new();
    private bool _disposed;

    public IDevice Device { get; }

    public CpuStream(IDevice device)
    {
        Device = device;
    }

    public void Enqueue(System.Action operation)
    {
        if (operation == null)
            throw new ArgumentNullException(nameof(operation));

        // CPU executes immediately (no true async)
        operation();
    }

    public IEvent RecordEvent()
    {
        // CPU events complete immediately
        return new CpuEvent(this, completed: true);
    }

    public void Wait(IEvent @event)
    {
        // CPU events are always complete, so this is a no-op
        if (@event == null)
            throw new ArgumentNullException(nameof(@event));
    }

    public void Synchronize()
    {
        // No pending operations on CPU
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
        }
    }
}
