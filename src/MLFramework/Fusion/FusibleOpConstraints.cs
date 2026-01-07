using MLFramework.Core;

namespace MLFramework.Fusion;

/// <summary>
/// Constraints for fusible operations
/// </summary>
public record FusibleOpConstraints
{
    /// <summary>
    /// Gets the required tensor layout
    /// </summary>
    public required TensorLayout RequiredLayout { get; init; }

    /// <summary>
    /// Gets the supported data types
    /// </summary>
    public required IReadOnlySet<DataType> SupportedDataTypes { get; init; }

    /// <summary>
    /// Gets or sets whether contiguous memory is required
    /// </summary>
    public bool RequiresContiguousMemory { get; init; } = true;

    /// <summary>
    /// Gets or sets the maximum fusion group size
    /// </summary>
    public int MaxFusionGroupSize { get; init; } = 16;

    /// <summary>
    /// Gets or sets whether fusion with in-place operations is supported
    /// </summary>
    public bool SupportsFusionWithInplaceOps { get; init; } = false;
}
