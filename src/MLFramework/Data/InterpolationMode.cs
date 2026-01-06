namespace MLFramework.Data;

/// <summary>
/// Interpolation modes for image resizing.
/// </summary>
public enum InterpolationMode
{
    /// <summary>
    /// Nearest-neighbor interpolation.
    /// </summary>
    Nearest,

    /// <summary>
    /// Bilinear interpolation.
    /// </summary>
    Bilinear,

    /// <summary>
    /// Bicubic interpolation.
    /// </summary>
    Bicubic,

    /// <summary>
    /// Lanczos resampling.
    /// </summary>
    Lanczos
}
