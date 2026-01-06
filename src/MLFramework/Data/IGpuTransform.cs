namespace MLFramework.Data;

/// <summary>
/// Interface for GPU-accelerated transforms.
/// </summary>
public interface IGpuTransform : ITransform
{
    /// <summary>
    /// Gets whether GPU acceleration is available.
    /// </summary>
    bool GpuAvailable { get; }

    /// <summary>
    /// Gets the current GPU device ID.
    /// </summary>
    int GpuDevice { get; }

    /// <summary>
    /// Sets the GPU device to use for the transform.
    /// </summary>
    /// <param name="deviceId">The GPU device ID.</param>
    void SetGpuDevice(int deviceId);
}
