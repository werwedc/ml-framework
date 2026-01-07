using System.Collections.ObjectModel;

namespace MLFramework.Shapes
{
    /// <summary>
    /// Represents a tensor shape using symbolic dimensions instead of concrete integers.
    /// This class is immutable to ensure thread-safety and enable safe sharing across operations.
    /// </summary>
    public sealed class SymbolicShape : IEquatable<SymbolicShape>
    {
        private readonly bool? _cachedIsFullyKnown;
        private readonly bool? _cachedIsPartiallyKnown;

        /// <summary>
        /// Gets the dimensions of this shape.
        /// </summary>
        public ReadOnlyCollection<SymbolicDimension> Dimensions { get; }

        /// <summary>
        /// Gets the rank (number of dimensions) of this shape.
        /// </summary>
        public int Rank => Dimensions.Count;

        /// <summary>
        /// Initializes a new instance of the SymbolicShape class.
        /// </summary>
        /// <param name="dimensions">The dimensions of the shape.</param>
        /// <exception cref="ArgumentNullException">Thrown when dimensions is null.</exception>
        public SymbolicShape(params SymbolicDimension[] dimensions)
        {
            Dimensions = new ReadOnlyCollection<SymbolicDimension>(
                dimensions ?? throw new ArgumentNullException(nameof(dimensions)));
        }

        /// <summary>
        /// Initializes a new instance of the SymbolicShape class with a collection of dimensions.
        /// </summary>
        /// <param name="dimensions">The dimensions of the shape.</param>
        /// <exception cref="ArgumentNullException">Thrown when dimensions is null.</exception>
        public SymbolicShape(IEnumerable<SymbolicDimension> dimensions)
        {
            Dimensions = new ReadOnlyCollection<SymbolicDimension>(
                dimensions?.ToArray() ?? throw new ArgumentNullException(nameof(dimensions)));
        }

        /// <summary>
        /// Gets the dimension at the specified index.
        /// Supports negative indexing from the end.
        /// </summary>
        /// <param name="index">The index of the dimension.</param>
        /// <returns>The dimension at the specified index.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when index is out of range.</exception>
        public SymbolicDimension GetDimension(int index)
        {
            // Handle negative indexing
            if (index < 0)
            {
                index += Rank;
            }

            if (index < 0 || index >= Rank)
            {
                throw new ArgumentOutOfRangeException(nameof(index),
                    $"Index {index} is out of range for shape with rank {Rank}");
            }

            return Dimensions[index];
        }

        /// <summary>
        /// Returns a new SymbolicShape with the specified dimension replaced.
        /// </summary>
        /// <param name="index">The index of the dimension to replace.</param>
        /// <param name="dim">The new dimension.</param>
        /// <returns>A new SymbolicShape with the updated dimension.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when index is out of range.</exception>
        /// <exception cref="ArgumentNullException">Thrown when dim is null.</exception>
        public SymbolicShape SetDimension(int index, SymbolicDimension dim)
        {
            if (dim == null)
                throw new ArgumentNullException(nameof(dim));

            // Handle negative indexing
            if (index < 0)
            {
                index += Rank;
            }

            if (index < 0 || index >= Rank)
            {
                throw new ArgumentOutOfRangeException(nameof(index),
                    $"Index {index} is out of range for shape with rank {Rank}");
            }

            var newDimensions = Dimensions.ToArray();
            newDimensions[index] = dim;
            return new SymbolicShape(newDimensions);
        }

        /// <summary>
        /// Returns true if all dimensions have concrete values.
        /// </summary>
        /// <returns>True if all dimensions are known; otherwise, false.</returns>
        public bool IsFullyKnown()
        {
            if (_cachedIsFullyKnown.HasValue)
                return _cachedIsFullyKnown.Value;

            return Dimensions.All(dim => dim.IsKnown());
        }

        /// <summary>
        /// Returns true if at least one dimension has a concrete value.
        /// </summary>
        /// <returns>True if at least one dimension is known; otherwise, false.</returns>
        public bool IsPartiallyKnown()
        {
            if (_cachedIsPartiallyKnown.HasValue)
                return _cachedIsPartiallyKnown.Value;

            return Dimensions.Any(dim => dim.IsKnown());
        }

        /// <summary>
        /// Converts this symbolic shape to a concrete integer array.
        /// </summary>
        /// <returns>An array of concrete dimension values.</returns>
        /// <exception cref="InvalidOperationException">Thrown when not all dimensions are known.</exception>
        public int[] ToConcrete()
        {
            if (!IsFullyKnown())
            {
                throw new InvalidOperationException(
                    "Cannot convert symbolic shape to concrete - not all dimensions are known. " +
                    $"Shape: {ToString()}");
            }

            return Dimensions.Select(dim => dim.Value!.Value).ToArray();
        }

        /// <summary>
        /// Creates a deep copy of this symbolic shape.
        /// </summary>
        /// <returns>A new SymbolicShape with cloned dimensions.</returns>
        public SymbolicShape Clone()
        {
            var clonedDimensions = Dimensions.Select(dim => dim.Clone()).ToArray();
            return new SymbolicShape(clonedDimensions);
        }

        /// <summary>
        /// Determines whether the specified SymbolicShape is equal to this instance.
        /// </summary>
        /// <param name="other">The SymbolicShape to compare with.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public bool Equals(SymbolicShape? other)
        {
            if (other is null)
                return false;

            if (ReferenceEquals(this, other))
                return true;

            if (Rank != other.Rank)
                return false;

            for (int i = 0; i < Rank; i++)
            {
                if (!Dimensions[i].Equals(other.Dimensions[i]))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Determines whether the specified object is equal to this instance.
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public override bool Equals(object? obj)
        {
            return Equals(obj as SymbolicShape);
        }

        /// <summary>
        /// Returns a hash code for this instance.
        /// </summary>
        /// <returns>A hash code based on dimensions.</returns>
        public override int GetHashCode()
        {
            var hash = new HashCode();
            foreach (var dimension in Dimensions)
            {
                hash.Add(dimension);
            }
            return hash.ToHashCode();
        }

        /// <summary>
        /// Returns a string representation of this symbolic shape.
        /// </summary>
        /// <returns>A string showing dimensions like "[batch_size, seq_len, 512]".</returns>
        public override string ToString()
        {
            return $"[{string.Join(", ", Dimensions.Select(dim => dim.ToString()))}]";
        }

        /// <summary>
        /// Determines whether two SymbolicShape instances are equal.
        /// </summary>
        /// <param name="left">The first instance.</param>
        /// <param name="right">The second instance.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public static bool operator ==(SymbolicShape? left, SymbolicShape? right)
        {
            if (left is null)
                return right is null;

            return left.Equals(right);
        }

        /// <summary>
        /// Determines whether two SymbolicShape instances are not equal.
        /// </summary>
        /// <param name="left">The first instance.</param>
        /// <param name="right">The second instance.</param>
        /// <returns>True if the instances are not equal; otherwise, false.</returns>
        public static bool operator !=(SymbolicShape? left, SymbolicShape? right)
        {
            return !(left == right);
        }
    }
}
