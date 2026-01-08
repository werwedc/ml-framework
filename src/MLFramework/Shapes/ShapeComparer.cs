namespace MLFramework.Shapes
{
    /// <summary>
    /// Provides static methods for comparing and manipulating symbolic shapes.
    /// </summary>
    public static class ShapeComparer
    {
        /// <summary>
        /// Determines if two shapes can be broadcast together according to numpy-style broadcasting rules.
        /// </summary>
        /// <param name="a">The first shape.</param>
        /// <param name="b">The second shape.</param>
        /// <returns>True if the shapes are compatible for broadcasting; otherwise, false.</returns>
        public static bool AreCompatible(SymbolicShape a, SymbolicShape b)
        {
            if (a == null || b == null)
                throw new ArgumentNullException(a == null ? nameof(a) : nameof(b));

            // Broadcast from right to left
            int maxRank = Math.Max(a.Rank, b.Rank);

            for (int i = 0; i < maxRank; i++)
            {
                var dimA = i < a.Rank ? a.GetDimension(-(i + 1)) : new SymbolicDimension($"_dummy_{i}", 1);
                var dimB = i < b.Rank ? b.GetDimension(-(i + 1)) : new SymbolicDimension($"_dummy_{i}", 1);

                // Check compatibility:
                // 1. Both are the same
                // 2. One is 1 (or symbolically 1)
                // 3. One is unknown
                if (!DimensionsAreBroadcastable(dimA, dimB))
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Determines if two dimensions can be broadcast together.
        /// </summary>
        /// <param name="a">The first dimension.</param>
        /// <param name="b">The second dimension.</param>
        /// <returns>True if the dimensions are broadcastable; otherwise, false.</returns>
        private static bool DimensionsAreBroadcastable(SymbolicDimension a, SymbolicDimension b)
        {
            // Same dimension - compatible
            if (a.Equals(b))
                return true;

            // One dimension is 1 - compatible
            if (a.IsKnown() && a.Value == 1)
                return true;
            if (b.IsKnown() && b.Value == 1)
                return true;

            // At least one dimension is unknown - assume compatible (runtime will validate)
            if (!a.IsKnown() || !b.IsKnown())
                return true;

            // Both are known and different - not compatible
            return false;
        }

        /// <summary>
        /// Computes the broadcast shape of two shapes.
        /// </summary>
        /// <param name="a">The first shape.</param>
        /// <param name="b">The second shape.</param>
        /// <returns>The broadcast shape.</returns>
        /// <exception cref="ArgumentException">Thrown when shapes are not compatible for broadcasting.</exception>
        public static SymbolicShape GetBroadcastShape(SymbolicShape a, SymbolicShape b)
        {
            if (a == null || b == null)
                throw new ArgumentNullException(a == null ? nameof(a) : nameof(b));

            if (!AreCompatible(a, b))
            {
                throw new ArgumentException(
                    $"Shapes {a} and {b} are not compatible for broadcasting");
            }

            int maxRank = Math.Max(a.Rank, b.Rank);
            var resultDims = new List<SymbolicDimension>();

            for (int i = 0; i < maxRank; i++)
            {
                var dimA = i < a.Rank ? a.GetDimension(-(i + 1)) : new SymbolicDimension($"_dummy_{i}", 1);
                var dimB = i < b.Rank ? b.GetDimension(-(i + 1)) : new SymbolicDimension($"_dummy_{i}", 1);

                resultDims.Insert(0, GetBroadcastDimension(dimA, dimB));
            }

            return new SymbolicShape(resultDims);
        }

        /// <summary>
        /// Computes the broadcast dimension for two dimensions.
        /// </summary>
        /// <param name="a">The first dimension.</param>
        /// <param name="b">The second dimension.</param>
        /// <returns>The broadcast dimension.</returns>
        private static SymbolicDimension GetBroadcastDimension(SymbolicDimension a, SymbolicDimension b)
        {
            // Same dimension
            if (a.Equals(b))
                return a;

            // One dimension is 1, take the other
            if (a.IsKnown() && a.Value == 1)
                return b;
            if (b.IsKnown() && b.Value == 1)
                return a;

            // Both are unknown - create a dimension with the broader bounds
            if (!a.IsKnown() && !b.IsKnown())
            {
                int newMin = Math.Max(a.MinValue, b.MinValue);
                int? newMax = a.MaxValue.HasValue && b.MaxValue.HasValue
                    ? Math.Max(a.MaxValue.Value, b.MaxValue.Value)
                    : null;

                return new SymbolicDimension(
                    $"{a.Name}_{b.Name}",
                    null,
                    newMin,
                    newMax);
            }

            // One is known, one is unknown - take the unknown (more general)
            return !a.IsKnown() ? a : b;
        }

        /// <summary>
        /// Determines if a reshape operation is valid.
        /// </summary>
        /// <param name="from">The source shape.</param>
        /// <param name="to">The target shape.</param>
        /// <returns>True if the reshape is valid; otherwise, false.</returns>
        public static bool CanReshape(SymbolicShape from, SymbolicShape to)
        {
            if (from == null || to == null)
                throw new ArgumentNullException(from == null ? nameof(from) : nameof(to));

            // If both are fully known, check total size equality
            if (from.IsFullyKnown() && to.IsFullyKnown())
            {
                long fromSize = from.ToConcrete().Aggregate(1L, (acc, dim) => acc * dim);
                long toSize = to.ToConcrete().Aggregate(1L, (acc, dim) => acc * dim);
                return fromSize == toSize;
            }

            // If both are not fully known, we can only validate structural constraints
            // Check that -1 (inferred dimension) appears at most once in target
            int inferredCount = 0;
            foreach (var dim in to.Dimensions)
            {
                if (!dim.IsKnown())
                    inferredCount++;
            }

            // Can only infer one dimension
            if (inferredCount > 1)
                return false;

            // Otherwise, assume valid (runtime will validate)
            return true;
        }

        /// <summary>
        /// Reshapes a shape to a new shape, inferring dimensions as needed.
        /// </summary>
        /// <param name="from">The source shape.</param>
        /// <param name="to">The target shape with potential -1 for inference.</param>
        /// <returns>The reshaped shape.</returns>
        /// <exception cref="ArgumentException">Thrown when reshape is invalid.</exception>
        public static SymbolicShape Reshape(SymbolicShape from, SymbolicShape to)
        {
            if (from == null || to == null)
                throw new ArgumentNullException(from == null ? nameof(from) : nameof(to));

            if (!CanReshape(from, to))
            {
                throw new ArgumentException(
                    $"Cannot reshape {from} to {to}");
            }

            // If both fully known, just return target
            if (from.IsFullyKnown() && to.IsFullyKnown())
                return to;

            // Handle dimension inference
            int inferredIndex = -1;
            long knownProduct = 1;

            for (int i = 0; i < to.Rank; i++)
            {
                var dim = to.GetDimension(i);
                if (dim.IsKnown())
                {
                    knownProduct *= dim.Value!.Value;
                }
                else if (inferredIndex == -1)
                {
                    inferredIndex = i;
                }
                else
                {
                    // More than one dimension to infer - not supported
                    throw new ArgumentException(
                        "Cannot reshape - only one dimension can be inferred (-1)");
                }
            }

            // If source is fully known, compute the inferred dimension
            if (from.IsFullyKnown() && inferredIndex >= 0)
            {
                long totalSize = from.ToConcrete().Aggregate(1L, (acc, dim) => acc * dim);
                if (totalSize % knownProduct != 0)
                {
                    throw new ArgumentException(
                        $"Cannot reshape - dimensions not divisible: {totalSize} / {knownProduct}");
                }

                int inferredValue = (int)(totalSize / knownProduct);
                var newDims = to.Dimensions.ToArray();
                newDims[inferredIndex] = new SymbolicDimension($"inferred_{inferredIndex}", inferredValue);
                return new SymbolicShape(newDims);
            }

            // Source not fully known - just return target
            return to;
        }
    }
}
