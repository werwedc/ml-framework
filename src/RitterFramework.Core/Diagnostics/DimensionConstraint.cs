namespace RitterFramework.Core.Diagnostics;

/// <summary>
/// Defines a constraint on a dimension of a tensor shape.
/// </summary>
public class DimensionConstraint
{
    /// <summary>
    /// Types of dimension constraints.
    /// </summary>
    public enum ConstraintType
    {
        /// <summary>Must match another dimension.</summary>
        MustMatch,

        /// <summary>Must equal a specific value.</summary>
        MustEqual,

        /// <summary>Must be greater than 0.</summary>
        MustBePositive,

        /// <summary>Must be multiple of a value.</summary>
        MustBeMultipleOf,

        /// <summary>Must divide another dimension evenly.</summary>
        MustDivide,

        /// <summary>Any value is acceptable.</summary>
        Any
    }

    /// <summary>
    /// The type of constraint.
    /// </summary>
    public ConstraintType Type { get; set; }

    /// <summary>
    /// Target input index (for MustMatch constraint).
    /// </summary>
    public int? TargetInputIndex { get; set; }

    /// <summary>
    /// Target dimension index (for MustMatch constraint).
    /// </summary>
    public int? TargetDimensionIndex { get; set; }

    /// <summary>
    /// Fixed value (for MustEqual constraint).
    /// </summary>
    public long? FixedValue { get; set; }

    /// <summary>
    /// Multiple of value (for MustBeMultipleOf constraint).
    /// </summary>
    public long? MultipleOf { get; set; }
}
