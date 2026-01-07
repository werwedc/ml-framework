namespace MLFramework.Fusion;

/// <summary>
/// Tensor layout enumeration for memory layout specification
/// </summary>
public enum TensorLayout
{
    /// <summary>Any layout (wildcard)</summary>
    Any,
    /// <summary>NCHW: Batch, Channels, Height, Width</summary>
    NCHW,
    /// <summary>NHWC: Batch, Height, Width, Channels</summary>
    NHWC,
    /// <summary>Contiguous/Flat layout</summary>
    Contiguous,
    /// <summary>Strided layout</summary>
    Strided
}
