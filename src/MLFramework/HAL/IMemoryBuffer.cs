namespace MLFramework.HAL;

/// <summary>
/// Represents a block of memory allocated on a device
/// </summary>
public interface IMemoryBuffer : IDisposable
{
    /// <summary>
    /// Pointer to the memory (unmanaged)
    /// </summary>
    IntPtr Pointer { get; }

    /// <summary>
    /// Size in bytes
    /// </summary>
    long Size { get; }

    /// <summary>
    /// Device this buffer is allocated on
    /// </summary>
    IDevice Device { get; }

    /// <summary>
    /// Check if buffer is valid (not disposed)
    /// </summary>
    bool IsValid { get; }
}
