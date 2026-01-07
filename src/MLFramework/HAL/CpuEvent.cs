namespace MLFramework.HAL;

/// <summary>
/// CPU event that is always complete (synchronous execution)
/// </summary>
public class CpuEvent : IEvent
{
    private bool _disposed;

    public IStream Stream { get; }
    public bool IsCompleted { get; private set; }

    public CpuEvent(IStream? stream, bool completed = false)
    {
        Stream = stream!;
        IsCompleted = completed;
    }

    public void Synchronize()
    {
        // Always complete for CPU
        IsCompleted = true;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
        }
    }
}
