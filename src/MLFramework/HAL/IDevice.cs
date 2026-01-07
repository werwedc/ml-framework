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
    /// <param name="size">Size in bytes to allocate</param>
    /// <returns>A memory buffer allocated on this device</returns>
    IMemoryBuffer AllocateMemory(long size);

    /// <summary>
    /// Free memory allocated on this device
    /// </summary>
    /// <param name="buffer">The buffer to free</param>
    void FreeMemory(IMemoryBuffer buffer);

    /// <summary>
    /// Create a compute stream for async operations
    /// </summary>
    /// <returns>A new stream for this device</returns>
    IStream CreateStream();

    /// <summary>
    /// Block until all operations on this device complete
    /// </summary>
    void Synchronize();

    /// <summary>
    /// Record an event at the current point in the default stream
    /// </summary>
    /// <returns>A new event representing the current stream position</returns>
    IEvent RecordEvent();
}
