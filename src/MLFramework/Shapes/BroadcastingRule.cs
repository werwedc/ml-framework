using System;

namespace MLFramework.Shapes
{
    /// <summary>
    /// Represents a rule for broadcasting two dimensions together.
    /// </summary>
    public sealed class BroadcastingRule : IEquatable<BroadcastingRule>
    {
        /// <summary>
        /// Gets the index of the dimension being broadcasted.
        /// </summary>
        public int DimensionIndex { get; }

        /// <summary>
        /// Gets whether the dimensions are broadcastable.
        /// </summary>
        public bool IsBroadcastable { get; }

        /// <summary>
        /// Gets the output size after broadcasting.
        /// </summary>
        public SymbolicDimension OutputSize { get; }

        /// <summary>
        /// Gets the first input dimension.
        /// </summary>
        public SymbolicDimension Dim1 { get; }

        /// <summary>
        /// Gets the second input dimension.
        /// </summary>
        public SymbolicDimension Dim2 { get; }

        /// <summary>
        /// Initializes a new instance of the BroadcastingRule class.
        /// </summary>
        /// <param name="dimensionIndex">The index of the dimension.</param>
        /// <param name="dim1">The first input dimension.</param>
        /// <param name="dim2">The second input dimension.</param>
        /// <param name="outputSize">The output size after broadcasting.</param>
        public BroadcastingRule(
            int dimensionIndex,
            SymbolicDimension dim1,
            SymbolicDimension dim2,
            SymbolicDimension outputSize)
        {
            DimensionIndex = dimensionIndex;
            Dim1 = dim1 ?? throw new ArgumentNullException(nameof(dim1));
            Dim2 = dim2 ?? throw new ArgumentNullException(nameof(dim2));
            OutputSize = outputSize ?? throw new ArgumentNullException(nameof(outputSize));
            IsBroadcastable = true;
        }

        /// <summary>
        /// Initializes a new instance of the BroadcastingRule class for non-broadcastable dimensions.
        /// </summary>
        /// <param name="dimensionIndex">The index of the dimension.</param>
        /// <param name="dim1">The first input dimension.</param>
        /// <param name="dim2">The second input dimension.</param>
        /// <param name="isBroadcastable">Whether the dimensions are broadcastable.</param>
        private BroadcastingRule(
            int dimensionIndex,
            SymbolicDimension dim1,
            SymbolicDimension dim2,
            bool isBroadcastable)
        {
            DimensionIndex = dimensionIndex;
            Dim1 = dim1 ?? throw new ArgumentNullException(nameof(dim1));
            Dim2 = dim2 ?? throw new ArgumentNullException(nameof(dim2));
            IsBroadcastable = isBroadcastable;
            OutputSize = null!;
        }

        /// <summary>
        /// Creates a non-broadcastable rule.
        /// </summary>
        /// <param name="dimensionIndex">The index of the dimension.</param>
        /// <param name="dim1">The first input dimension.</param>
        /// <param name="dim2">The second input dimension.</param>
        /// <returns>A non-broadcastable rule.</returns>
        public static BroadcastingRule CreateNonBroadcastable(
            int dimensionIndex,
            SymbolicDimension dim1,
            SymbolicDimension dim2)
        {
            return new BroadcastingRule(dimensionIndex, dim1, dim2, false);
        }

        /// <summary>
        /// Applies the broadcasting rule to two dimensions and returns the output dimension.
        /// </summary>
        /// <param name="dim1">The first dimension.</param>
        /// <param name="dim2">The second dimension.</param>
        /// <returns>The output dimension after broadcasting.</returns>
        public static SymbolicDimension Apply(SymbolicDimension dim1, SymbolicDimension dim2)
        {
            if (dim1 == null)
                throw new ArgumentNullException(nameof(dim1));
            if (dim2 == null)
                throw new ArgumentNullException(nameof(dim2));

            // If dimensions are equal, output is that dimension
            if (dim1.Value.HasValue && dim2.Value.HasValue && dim1.Value == dim2.Value)
            {
                return dim1;
            }

            // If one dimension is 1, output is the other dimension
            if (dim1.Value == 1 && dim2.Value != 1)
            {
                return dim2;
            }

            if (dim2.Value == 1 && dim1.Value != 1)
            {
                return dim1;
            }

            // If both are 1, output is 1
            if (dim1.Value == 1 && dim2.Value == 1)
            {
                return dim1;
            }

            // If one dimension is unknown (symbolic), output is symbolic with constraint >= 1
            if (!dim1.Value.HasValue)
            {
                // Add constraint that dimension >= 1
                return dim1.WithConstraints(1, dim1.MaxValue);
            }

            if (!dim2.Value.HasValue)
            {
                // Add constraint that dimension >= 1
                return dim2.WithConstraints(1, dim2.MaxValue);
            }

            // Both are concrete but not equal and neither is 1 - incompatible
            // This shouldn't happen if CanBroadcast was called first
            throw new InvalidOperationException(
                $"Cannot broadcast dimensions {dim1} and {dim2}");
        }

        /// <summary>
        /// Determines whether the specified BroadcastingRule is equal to this instance.
        /// </summary>
        /// <param name="other">The BroadcastingRule to compare with.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public bool Equals(BroadcastingRule? other)
        {
            if (other is null)
                return false;

            if (ReferenceEquals(this, other))
                return true;

            return DimensionIndex == other.DimensionIndex &&
                   IsBroadcastable == other.IsBroadcastable &&
                   OutputSize.Equals(other.OutputSize) &&
                   Dim1.Equals(other.Dim1) &&
                   Dim2.Equals(other.Dim2);
        }

        /// <summary>
        /// Determines whether the specified object is equal to this instance.
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public override bool Equals(object? obj)
        {
            return Equals(obj as BroadcastingRule);
        }

        /// <summary>
        /// Returns a hash code for this instance.
        /// </summary>
        /// <returns>A hash code based on properties.</returns>
        public override int GetHashCode()
        {
            var hash = new HashCode();
            hash.Add(DimensionIndex);
            hash.Add(IsBroadcastable);
            hash.Add(OutputSize);
            hash.Add(Dim1);
            hash.Add(Dim2);
            return hash.ToHashCode();
        }

        /// <summary>
        /// Returns a string representation of this broadcasting rule.
        /// </summary>
        /// <returns>A string describing the rule.</returns>
        public override string ToString()
        {
            if (!IsBroadcastable)
            {
                return $"Non-broadcastable at index {DimensionIndex}: {Dim1} vs {Dim2}";
            }

            return $"Broadcast at index {DimensionIndex}: {Dim1} and {Dim2} -> {OutputSize}";
        }

        /// <summary>
        /// Determines whether two BroadcastingRule instances are equal.
        /// </summary>
        /// <param name="left">The first instance.</param>
        /// <param name="right">The second instance.</param>
        /// <returns>True if the instances are equal; otherwise, false.</returns>
        public static bool operator ==(BroadcastingRule? left, BroadcastingRule? right)
        {
            if (left is null)
                return right is null;

            return left.Equals(right);
        }

        /// <summary>
        /// Determines whether two BroadcastingRule instances are not equal.
        /// </summary>
        /// <param name="left">The first instance.</param>
        /// <param name="right">The second instance.</param>
        /// <returns>True if the instances are not equal; otherwise, false.</returns>
        public static bool operator !=(BroadcastingRule? left, BroadcastingRule? right)
        {
            return !(left == right);
        }
    }
}
