using System;

namespace MLFramework.Shapes
{
    /// <summary>
    /// Represents a constraint that requires a dimension value to be within a specified range.
    /// </summary>
    public sealed class RangeConstraint : IShapeConstraint, IEquatable<RangeConstraint>
    {
        /// <summary>
        /// Gets the minimum value (inclusive) for this constraint.
        /// </summary>
        public int MinValue { get; }

        /// <summary>
        /// Gets the maximum value (inclusive) for this constraint.
        /// </summary>
        public int MaxValue { get; }

        /// <summary>
        /// Initializes a new instance of the RangeConstraint class.
        /// </summary>
        /// <param name="minValue">The minimum value (inclusive).</param>
        /// <param name="maxValue">The maximum value (inclusive).</param>
        /// <exception cref="ArgumentException">Thrown when minValue > maxValue.</exception>
        public RangeConstraint(int minValue, int maxValue)
        {
            if (minValue < 0)
                throw new ArgumentException("MinValue must be non-negative.", nameof(minValue));

            if (maxValue < minValue)
                throw new ArgumentException("MaxValue must be greater than or equal to MinValue.", nameof(maxValue));

            MinValue = minValue;
            MaxValue = maxValue;
        }

        /// <summary>
        /// Checks if the constraint is satisfied by the given symbolic dimension.
        /// </summary>
        /// <param name="dim">The symbolic dimension to validate.</param>
        /// <returns>True if the dimension value is within the range; otherwise, false.</returns>
        public bool Validate(SymbolicDimension dim)
        {
            if (dim == null)
                return false;

            if (!dim.Value.HasValue)
                return false;

            int value = dim.Value.Value;
            return value >= MinValue && value <= MaxValue;
        }

        /// <summary>
        /// Forces a value into the valid range by clamping.
        /// </summary>
        /// <param name="value">The value to clamp.</param>
        /// <returns>The clamped value within the valid range.</returns>
        public int Clamp(int value)
        {
            if (value < MinValue)
                return MinValue;

            if (value > MaxValue)
                return MaxValue;

            return value;
        }

        /// <summary>
        /// Returns a human-readable description of this constraint.
        /// </summary>
        /// <returns>A string describing the range constraint.</returns>
        public override string ToString()
        {
            return $"Range [{MinValue}, {MaxValue}]";
        }

        /// <summary>
        /// Determines whether the specified RangeConstraint is equal to this instance.
        /// </summary>
        /// <param name="other">The RangeConstraint to compare with.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public bool Equals(RangeConstraint? other)
        {
            if (other is null)
                return false;

            if (ReferenceEquals(this, other))
                return true;

            return MinValue == other.MinValue && MaxValue == other.MaxValue;
        }

        /// <summary>
        /// Determines whether the specified object is equal to this instance.
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public override bool Equals(object? obj)
        {
            return Equals(obj as RangeConstraint);
        }

        /// <summary>
        /// Returns a hash code for this instance.
        /// </summary>
        /// <returns>A hash code based on min and max values.</returns>
        public override int GetHashCode()
        {
            var hash = new HashCode();
            hash.Add(MinValue);
            hash.Add(MaxValue);
            return hash.ToHashCode();
        }

        /// <summary>
        /// Determines whether two RangeConstraint instances are equal.
        /// </summary>
        /// <param name="left">The first instance.</param>
        /// <param name="right">The second instance.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public static bool operator ==(RangeConstraint? left, RangeConstraint? right)
        {
            if (left is null)
                return right is null;

            return left.Equals(right);
        }

        /// <summary>
        /// Determines whether two RangeConstraint instances are not equal.
        /// </summary>
        /// <param name="left">The first instance.</param>
        /// <param name="right">The second instance.</param>
        /// <returns>True if the instances are not equal; otherwise, false.</returns>
        public static bool operator !=(RangeConstraint? left, RangeConstraint? right)
        {
            return !(left == right);
        }
    }
}
