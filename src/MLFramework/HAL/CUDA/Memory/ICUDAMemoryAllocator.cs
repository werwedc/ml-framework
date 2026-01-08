namespace MLFramework.HAL.CUDA;

/// <summary>
/// Interface for CUDA memory allocators with graph support.
/// </summary>
public interface ICUDAMemoryAllocator : IDisposable
{
    /// <summary>
    /// Allocates GPU memory.
    /// </summary>
    /// <param name="size">Size in bytes to allocate</param>
    /// <param name="alignment">Alignment requirement in bytes (default: 256)</param>
    /// <returns>Pointer to the allocated memory</returns>
    IntPtr Allocate(ulong size, ulong alignment = 256);

    /// <summary>
    /// Frees GPU memory.
    /// </summary>
    /// <param name="ptr">Pointer to the memory to free</param>
    void Free(IntPtr ptr);

    /// <summary>
    /// Gets or sets the graph memory pool for graph-compatible allocations.
    /// </summary>
    CUDAGraphMemoryPool? GraphPool { get; set; }

    /// <summary>
    /// Gets whether graph mode is enabled.
    /// </summary>
    bool IsGraphMode { get; }

    /// <summary>
    /// Enables or disables graph mode.
    /// </summary>
    /// <param name="enabled">True to enable graph mode, false to disable</param>
    void SetGraphMode(bool enabled);
}
