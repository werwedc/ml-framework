namespace MLFramework.HAL;

/// <summary>
/// Represents a compute device (CPU, GPU, etc.)
/// </summary>
public interface IDevice
{
    /// <summary>
    /// Device type identifier (CPU, CUDA, ROCm, etc.)
    /// </summary>
    DeviceType DeviceType { get; }

    /// <summary>
    /// Unique device ID within the device type
    /// </summary>
    int DeviceId { get; }

    /// <summary>
    /// Allocate memory on this device
    /// </summary>
    IMemoryBuffer AllocateMemory(long size);

    /// <summary>
    /// Free memory allocated on this device
    /// </summary>
    void FreeMemory(IMemoryBuffer buffer);

    /// <summary>
    /// Create a compute stream for async operations
    /// </summary>
    IStream CreateStream();

    /// <summary>
    /// Block until all operations on this device complete
    /// </summary>
    void Synchronize();

    /// <summary>
    /// Record an event at the current point in the default stream
    /// </summary>
    IEvent RecordEvent();
}
