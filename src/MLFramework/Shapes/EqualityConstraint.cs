using System;

namespace MLFramework.Shapes
{
    /// <summary>
    /// Represents a constraint that requires a dimension value to be exactly equal to a target value.
    /// </summary>
    public sealed class EqualityConstraint : IShapeConstraint, IEquatable<EqualityConstraint>
    {
        /// <summary>
        /// Gets the target value that the dimension must equal.
        /// </summary>
        public int TargetValue { get; }

        /// <summary>
        /// Initializes a new instance of the EqualityConstraint class.
        /// </summary>
        /// <param name="targetValue">The target value that the dimension must equal.</param>
        /// <exception cref="ArgumentException">Thrown when targetValue is negative.</exception>
        public EqualityConstraint(int targetValue)
        {
            if (targetValue < 0)
                throw new ArgumentException("TargetValue must be non-negative.", nameof(targetValue));

            TargetValue = targetValue;
        }

        /// <summary>
        /// Checks if the constraint is satisfied by the given symbolic dimension.
        /// </summary>
        /// <param name="dim">The symbolic dimension to validate.</param>
        /// <returns>True if the dimension value equals the target; otherwise, false.</returns>
        public bool Validate(SymbolicDimension dim)
        {
            if (dim == null)
                return false;

            if (!dim.Value.HasValue)
                return false;

            return dim.Value.Value == TargetValue;
        }

        /// <summary>
        /// Returns a human-readable description of this constraint.
        /// </summary>
        /// <returns>A string describing the equality constraint.</returns>
        public override string ToString()
        {
            return $"Equals {TargetValue}";
        }

        /// <summary>
        /// Determines whether the specified EqualityConstraint is equal to this instance.
        /// </summary>
        /// <param name="other">The EqualityConstraint to compare with.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public bool Equals(EqualityConstraint? other)
        {
            if (other is null)
                return false;

            if (ReferenceEquals(this, other))
                return true;

            return TargetValue == other.TargetValue;
        }

        /// <summary>
        /// Determines whether the specified object is equal to this instance.
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public override bool Equals(object? obj)
        {
            return Equals(obj as EqualityConstraint);
        }

        /// <summary>
        /// Returns a hash code for this instance.
        /// </summary>
        /// <returns>A hash code based on the target value.</returns>
        public override int GetHashCode()
        {
            return HashCode.Combine(TargetValue);
        }

        /// <summary>
        /// Determines whether two EqualityConstraint instances are equal.
        /// </summary>
        /// <param name="left">The first instance.</param>
        /// <param name="right">The second instance.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public static bool operator ==(EqualityConstraint? left, EqualityConstraint? right)
        {
            if (left is null)
                return right is null;

            return left.Equals(right);
        }

        /// <summary>
        /// Determines whether two EqualityConstraint instances are not equal.
        /// </summary>
        /// <param name="left">The first instance.</param>
        /// <param name="right">The second instance.</param>
        /// <returns>True if the instances are not equal; otherwise, false.</returns>
        public static bool operator !=(EqualityConstraint? left, EqualityConstraint? right)
        {
            return !(left == right);
        }
    }
}
