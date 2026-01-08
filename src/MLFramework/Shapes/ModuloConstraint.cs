using System;

namespace MLFramework.Shapes
{
    /// <summary>
    /// Represents a constraint that requires a dimension value to be divisible by a specified divisor.
    /// </summary>
    public sealed class ModuloConstraint : IShapeConstraint, IEquatable<ModuloConstraint>
    {
        /// <summary>
        /// Gets the divisor for this constraint.
        /// </summary>
        public int Divisor { get; }

        /// <summary>
        /// Initializes a new instance of the ModuloConstraint class.
        /// </summary>
        /// <param name="divisor">The divisor that the dimension value must be divisible by.</param>
        /// <exception cref="ArgumentException">Thrown when divisor is less than or equal to zero.</exception>
        public ModuloConstraint(int divisor)
        {
            if (divisor <= 0)
                throw new ArgumentException("Divisor must be greater than zero.", nameof(divisor));

            Divisor = divisor;
        }

        /// <summary>
        /// Checks if the constraint is satisfied by the given symbolic dimension.
        /// </summary>
        /// <param name="dim">The symbolic dimension to validate.</param>
        /// <returns>True if the dimension value is divisible by the divisor; otherwise, false.</returns>
        public bool Validate(SymbolicDimension dim)
        {
            if (dim == null)
                return false;

            if (!dim.Value.HasValue)
                return false;

            return dim.Value.Value % Divisor == 0;
        }

        /// <summary>
        /// Returns a human-readable description of this constraint.
        /// </summary>
        /// <returns>A string describing the modulo constraint.</returns>
        public override string ToString()
        {
            return $"Modulo {Divisor}";
        }

        /// <summary>
        /// Determines whether the specified ModuloConstraint is equal to this instance.
        /// </summary>
        /// <param name="other">The ModuloConstraint to compare with.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public bool Equals(ModuloConstraint? other)
        {
            if (other is null)
                return false;

            if (ReferenceEquals(this, other))
                return true;

            return Divisor == other.Divisor;
        }

        /// <summary>
        /// Determines whether the specified object is equal to this instance.
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public override bool Equals(object? obj)
        {
            return Equals(obj as ModuloConstraint);
        }

        /// <summary>
        /// Returns a hash code for this instance.
        /// </summary>
        /// <returns>A hash code based on the divisor.</returns>
        public override int GetHashCode()
        {
            return HashCode.Combine(Divisor);
        }

        /// <summary>
        /// Determines whether two ModuloConstraint instances are equal.
        /// </summary>
        /// <param name="left">The first instance.</param>
        /// <param name="right">The second instance.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public static bool operator ==(ModuloConstraint? left, ModuloConstraint? right)
        {
            if (left is null)
                return right is null;

            return left.Equals(right);
        }

        /// <summary>
        /// Determines whether two ModuloConstraint instances are not equal.
        /// </summary>
        /// <param name="left">The first instance.</param>
        /// <param name="right">The second instance.</param>
        /// <returns>True if the instances are not equal; otherwise, false.</returns>
        public static bool operator !=(ModuloConstraint? left, ModuloConstraint? right)
        {
            return !(left == right);
        }
    }
}
