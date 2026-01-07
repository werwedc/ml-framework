namespace MLFramework.Distributed.Communication;

/// <summary>
/// Reduction operations for distributed communication.
/// </summary>
public enum ReduceOperation
{
    /// <summary>
    /// Sum of all values.
    /// </summary>
    Sum,

    /// <summary>
    /// Maximum of all values.
    /// </summary>
    Max,

    /// <summary>
    /// Minimum of all values.
    /// </summary>
    Min,

    /// <summary>
    /// Product of all values.
    /// </summary>
    Product,

    /// <summary>
    /// Average of all values.
    /// </summary>
    Avg
}
