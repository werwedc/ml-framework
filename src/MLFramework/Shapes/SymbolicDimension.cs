using System;

namespace MLFramework.Shapes
{
    /// <summary>
    /// Represents a symbolic tensor dimension that can be unknown, partially known, or fully known.
    /// This class is immutable to ensure thread-safety and enable safe sharing across operations.
    /// </summary>
    public sealed class SymbolicDimension : IEquatable<SymbolicDimension>
    {
        /// <summary>
        /// Gets the name of this symbolic dimension (e.g., "batch_size", "seq_len").
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Gets the concrete value if known, or null if the dimension is truly unknown.
        /// </summary>
        public int? Value { get; }

        /// <summary>
        /// Gets the minimum value for this dimension (lower bound). Defaults to 0.
        /// </summary>
        public int MinValue { get; }

        /// <summary>
        /// Gets the maximum value for this dimension (upper bound), or null if unbounded.
        /// </summary>
        public int? MaxValue { get; }

        /// <summary>
        /// Initializes a new instance of the SymbolicDimension class.
        /// </summary>
        /// <param name="name">The name of the dimension.</param>
        /// <param name="value">The concrete value if known, or null.</param>
        /// <param name="minValue">The minimum value (lower bound). Defaults to 0.</param>
        /// <param name="maxValue">The maximum value (upper bound), or null for unbounded.</param>
        /// <exception cref="ArgumentNullException">Thrown when name is null.</exception>
        /// <exception cref="ArgumentException">Thrown when bounds are invalid.</exception>
        public SymbolicDimension(string name, int? value = null, int minValue = 0, int? maxValue = null)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentNullException(nameof(name));

            if (minValue < 0)
                throw new ArgumentException("MinValue must be non-negative.", nameof(minValue));

            if (maxValue.HasValue && maxValue < minValue)
                throw new ArgumentException("MaxValue must be greater than or equal to MinValue.", nameof(maxValue));

            if (value.HasValue && (value.Value < minValue || (maxValue.HasValue && value.Value > maxValue.Value)))
                throw new ArgumentException("Value must be within MinValue and MaxValue bounds.", nameof(value));

            Name = name;
            Value = value;
            MinValue = minValue;
            MaxValue = maxValue;
        }

        /// <summary>
        /// Returns true if the dimension has a concrete value.
        /// </summary>
        public bool IsKnown() => Value.HasValue;

        /// <summary>
        /// Returns true if the dimension has a maximum value bound set.
        /// </summary>
        public bool IsBounded() => MaxValue.HasValue;

        /// <summary>
        /// Creates a shallow copy of this symbolic dimension.
        /// </summary>
        /// <returns>A new SymbolicDimension instance with the same properties.</returns>
        public SymbolicDimension Clone()
        {
            return new SymbolicDimension(Name, Value, MinValue, MaxValue);
        }

        /// <summary>
        /// Returns a new SymbolicDimension instance with the specified constraints.
        /// </summary>
        /// <param name="min">The new minimum value, or null to keep the current minimum.</param>
        /// <param name="max">The new maximum value, or null to keep the current maximum.</param>
        /// <returns>A new SymbolicDimension with the updated constraints.</returns>
        /// <exception cref="ArgumentException">Thrown when the new constraints are invalid.</exception>
        public SymbolicDimension WithConstraints(int? min, int? max)
        {
            var newMin = min ?? MinValue;
            var newMax = max ?? MaxValue;

            return new SymbolicDimension(Name, Value, newMin, newMax);
        }

        /// <summary>
        /// Determines whether the specified SymbolicDimension is equal to this instance.
        /// Two dimensions are equal if they have the same name, value, and bounds.
        /// </summary>
        /// <param name="other">The SymbolicDimension to compare with.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public bool Equals(SymbolicDimension? other)
        {
            if (other is null)
                return false;

            if (ReferenceEquals(this, other))
                return true;

            return Name == other.Name &&
                   Value == other.Value &&
                   MinValue == other.MinValue &&
                   MaxValue == other.MaxValue;
        }

        /// <summary>
        /// Determines whether the specified object is equal to this instance.
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public override bool Equals(object? obj)
        {
            return Equals(obj as SymbolicDimension);
        }

        /// <summary>
        /// Returns a hash code for this instance.
        /// </summary>
        /// <returns>A hash code based on name, value, and bounds.</returns>
        public override int GetHashCode()
        {
            var hash = new HashCode();
            hash.Add(Name);
            hash.Add(Value);
            hash.Add(MinValue);
            hash.Add(MaxValue);
            return hash.ToHashCode();
        }

        /// <summary>
        /// Returns a string representation of this symbolic dimension.
        /// </summary>
        /// <returns>A string showing the dimension name and its state.</returns>
        public override string ToString()
        {
            if (Value.HasValue)
                return $"{Name}={Value}";

            if (MaxValue.HasValue)
                return $"{Name}[{MinValue}..{MaxValue}]";

            return $"{Name}[{MinValue}..∞]";
        }

        /// <summary>
        /// Determines whether two SymbolicDimension instances are equal.
        /// </summary>
        /// <param name="left">The first instance.</param>
        /// <param name="right">The second instance.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public static bool operator ==(SymbolicDimension? left, SymbolicDimension? right)
        {
            if (left is null)
                return right is null;

            return left.Equals(right);
        }

        /// <summary>
        /// Determines whether two SymbolicDimension instances are not equal.
        /// </summary>
        /// <param name="left">The first instance.</param>
        /// <param name="right">The second instance.</param>
        /// <returns>True if the instances are not equal; otherwise, false.</returns>
        public static bool operator !=(SymbolicDimension? left, SymbolicDimension? right)
        {
            return !(left == right);
        }
    }
}
