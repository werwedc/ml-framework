namespace MLFramework.HAL;

/// <summary>
/// Enumeration of supported device types
/// </summary>
public enum DeviceType
{
    /// <summary>
    /// CPU device (default fallback)
    /// </summary>
    CPU,

    /// <summary>
    /// NVIDIA CUDA GPU
    /// </summary>
    CUDA,

    /// <summary>
    /// AMD ROCm GPU
    /// </summary>
    ROCm,

    /// <summary>
    /// Apple Metal GPU
    /// </summary>
    Metal,

    /// <summary>
    /// Intel oneAPI GPU
    /// </summary>
    OneAPI
}
