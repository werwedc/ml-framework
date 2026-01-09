namespace MLFramework.Diagnostics
{
    /// <summary>
    /// Represents a constraint on a tensor dimension used for shape validation.
    /// </summary>
    public class DimensionConstraint
    {
        /// <summary>
        /// The type of constraint to apply to the dimension.
        /// </summary>
        public enum ConstraintType
        {
            /// <summary>
            /// Must match another dimension from a different input tensor.
            /// </summary>
            MustMatch,

            /// <summary>
            /// Must equal a specific fixed value.
            /// </summary>
            MustEqual,

            /// <summary>
            /// Must be greater than zero.
            /// </summary>
            MustBePositive,

            /// <summary>
            /// Must be a multiple of a specific value.
            /// </summary>
            MustBeMultipleOf,

            /// <summary>
            /// Must divide another dimension evenly.
            /// </summary>
            MustDivide,

            /// <summary>
            /// Any value is acceptable (no constraint).
            /// </summary>
            Any
        }

        /// <summary>
        /// Gets or sets the type of constraint to apply.
        /// </summary>
        public ConstraintType Type { get; set; }

        /// <summary>
        /// Gets or sets the index of the input tensor whose dimension should be matched.
        /// Used when Type is MustMatch.
        /// </summary>
        public int? TargetInputIndex { get; set; }

        /// <summary>
        /// Gets or sets the dimension index to match within the target input tensor.
        /// Used when Type is MustMatch.
        /// </summary>
        public int? TargetDimensionIndex { get; set; }

        /// <summary>
        /// Gets or sets the fixed value the dimension must equal.
        /// Used when Type is MustEqual.
        /// </summary>
        public long? FixedValue { get; set; }

        /// <summary>
        /// Gets or sets the value the dimension must be a multiple of.
        /// Used when Type is MustBeMultipleOf.
        /// </summary>
        public long? MultipleOf { get; set; }

        /// <summary>
        /// Initializes a new instance of DimensionConstraint with default values.
        /// </summary>
        public DimensionConstraint()
        {
            Type = ConstraintType.Any;
        }

        /// <summary>
        /// Validates a dimension value against this constraint.
        /// </summary>
        /// <param name="value">The dimension value to validate.</param>
        /// <param name="targetValue">The target dimension value (for MustMatch or MustDivide constraints).</param>
        /// <returns>True if the constraint is satisfied, false otherwise.</returns>
        public bool Validate(long value, long? targetValue = null)
        {
            switch (Type)
            {
                case ConstraintType.Any:
                    return true;

                case ConstraintType.MustEqual:
                    if (!FixedValue.HasValue)
                        return false;
                    return value == FixedValue.Value;

                case ConstraintType.MustBePositive:
                    return value > 0;

                case ConstraintType.MustBeMultipleOf:
                    if (!MultipleOf.HasValue || MultipleOf.Value == 0)
                        return false;
                    return value % MultipleOf.Value == 0;

                case ConstraintType.MustMatch:
                    if (!targetValue.HasValue)
                        return false;
                    return value == targetValue.Value;

                case ConstraintType.MustDivide:
                    if (!targetValue.HasValue || value == 0)
                        return false;
                    return targetValue.Value % value == 0;

                default:
                    return false;
            }
        }

        /// <summary>
        /// Creates a MustMatch constraint.
        /// </summary>
        /// <param name="targetInputIndex">The index of the input tensor to match.</param>
        /// <param name="targetDimensionIndex">The dimension index to match.</param>
        /// <returns>A DimensionConstraint configured for MustMatch.</returns>
        public static DimensionConstraint CreateMustMatch(int targetInputIndex, int targetDimensionIndex)
        {
            return new DimensionConstraint
            {
                Type = ConstraintType.MustMatch,
                TargetInputIndex = targetInputIndex,
                TargetDimensionIndex = targetDimensionIndex
            };
        }

        /// <summary>
        /// Creates a MustEqual constraint.
        /// </summary>
        /// <param name="fixedValue">The fixed value the dimension must equal.</param>
        /// <returns>A DimensionConstraint configured for MustEqual.</returns>
        public static DimensionConstraint CreateMustEqual(long fixedValue)
        {
            return new DimensionConstraint
            {
                Type = ConstraintType.MustEqual,
                FixedValue = fixedValue
            };
        }

        /// <summary>
        /// Creates a MustBePositive constraint.
        /// </summary>
        /// <returns>A DimensionConstraint configured for MustBePositive.</returns>
        public static DimensionConstraint CreateMustBePositive()
        {
            return new DimensionConstraint
            {
                Type = ConstraintType.MustBePositive
            };
        }

        /// <summary>
        /// Creates a MustBeMultipleOf constraint.
        /// </summary>
        /// <param name="multipleOf">The value the dimension must be a multiple of.</param>
        /// <returns>A DimensionConstraint configured for MustBeMultipleOf.</returns>
        public static DimensionConstraint CreateMustBeMultipleOf(long multipleOf)
        {
            return new DimensionConstraint
            {
                Type = ConstraintType.MustBeMultipleOf,
                MultipleOf = multipleOf
            };
        }

        /// <summary>
        /// Creates a MustDivide constraint.
        /// </summary>
        /// <returns>A DimensionConstraint configured for MustDivide.</returns>
        public static DimensionConstraint CreateMustDivide()
        {
            return new DimensionConstraint
            {
                Type = ConstraintType.MustDivide
            };
        }

        /// <summary>
        /// Creates an Any constraint (no restriction).
        /// </summary>
        /// <returns>A DimensionConstraint configured for Any.</returns>
        public static DimensionConstraint CreateAny()
        {
            return new DimensionConstraint
            {
                Type = ConstraintType.Any
            };
        }

        /// <summary>
        /// Returns a string representation of the constraint.
        /// </summary>
        /// <returns>A description of the constraint.</returns>
        public override string ToString()
        {
            switch (Type)
            {
                case ConstraintType.MustMatch:
                    return $"MustMatch(input[{TargetInputIndex}], dim[{TargetDimensionIndex}])";
                case ConstraintType.MustEqual:
                    return $"MustEqual({FixedValue})";
                case ConstraintType.MustBePositive:
                    return "MustBePositive";
                case ConstraintType.MustBeMultipleOf:
                    return $"MustBeMultipleOf({MultipleOf})";
                case ConstraintType.MustDivide:
                    return "MustDivide";
                case ConstraintType.Any:
                    return "Any";
                default:
                    return "Unknown";
            }
        }
    }
}
