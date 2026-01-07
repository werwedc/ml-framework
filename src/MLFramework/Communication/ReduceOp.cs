namespace MLFramework.Communication;

using RitterFramework.Core.Tensor;

/// <summary>
/// Reduce operations for collective communication
/// </summary>
public enum ReduceOp
{
    /// <summary>
    /// Sum of all values
    /// </summary>
    Sum,

    /// <summary>
    /// Product of all values
    /// </summary>
    Product,

    /// <summary>
    /// Maximum of all values
    /// </summary>
    Max,

    /// <summary>
    /// Minimum of all values
    /// </summary>
    Min,

    /// <summary>
    /// Average of all values
    /// </summary>
    Avg
}
