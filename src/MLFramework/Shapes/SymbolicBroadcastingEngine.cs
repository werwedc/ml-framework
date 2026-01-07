using System.Collections.Generic;

namespace MLFramework.Shapes
{
    /// <summary>
    /// Engine for determining broadcasting rules and output shapes with symbolic dimensions.
    /// Follows NumPy broadcasting semantics.
    /// </summary>
    public class SymbolicBroadcastingEngine
    {
        // Cache for broadcast plans to improve performance
        private readonly Dictionary<(int, int), List<BroadcastingRule>> _broadcastPlanCache;

        /// <summary>
        /// Initializes a new instance of the SymbolicBroadcastingEngine class.
        /// </summary>
        public SymbolicBroadcastingEngine()
        {
            _broadcastPlanCache = new Dictionary<(int, int), List<BroadcastingRule>>();
        }

        /// <summary>
        /// Determines if two shapes can be broadcast together.
        /// </summary>
        /// <param name="shape1">The first shape.</param>
        /// <param name="shape2">The second shape.</param>
        /// <returns>True if the shapes can be broadcast; otherwise, false.</returns>
        public bool CanBroadcast(SymbolicShape shape1, SymbolicShape shape2)
        {
            if (shape1 == null)
                throw new ArgumentNullException(nameof(shape1));
            if (shape2 == null)
                throw new ArgumentNullException(nameof(shape2));

            int rank1 = shape1.Rank;
            int rank2 = shape2.Rank;
            int maxRank = Math.Max(rank1, rank2);

            // Align shapes from right to left
            for (int i = 1; i <= maxRank; i++)
            {
                int idx1 = rank1 - i;
                int idx2 = rank2 - i;

                SymbolicDimension dim1 = idx1 >= 0 ? shape1.GetDimension(idx1) : SymbolicDimensionFactory.CreateKnown("dim_implicit_1", 1);
                SymbolicDimension dim2 = idx2 >= 0 ? shape2.GetDimension(idx2) : SymbolicDimensionFactory.CreateKnown("dim_implicit_1", 1);

                if (!AreDimensionsBroadcastable(dim1, dim2))
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Computes the broadcast shape for two input shapes.
        /// </summary>
        /// <param name="shape1">The first shape.</param>
        /// <param name="shape2">The second shape.</param>
        /// <returns>The broadcast shape.</returns>
        /// <exception cref="InvalidOperationException">Thrown when shapes cannot be broadcast.</exception>
        public SymbolicShape GetBroadcastShape(SymbolicShape shape1, SymbolicShape shape2)
        {
            if (!CanBroadcast(shape1, shape2))
            {
                throw new InvalidOperationException(
                    $"Cannot broadcast shapes {shape1} and {shape2}");
            }

            int rank1 = shape1.Rank;
            int rank2 = shape2.Rank;
            int maxRank = Math.Max(rank1, rank2);

            List<SymbolicDimension> outputDims = new List<SymbolicDimension>(maxRank);

            // Align shapes from right to left
            for (int i = 1; i <= maxRank; i++)
            {
                int idx1 = rank1 - i;
                int idx2 = rank2 - i;

                SymbolicDimension dim1 = idx1 >= 0 ? shape1.GetDimension(idx1) : SymbolicDimensionFactory.CreateKnown("dim_implicit_1", 1);
                SymbolicDimension dim2 = idx2 >= 0 ? shape2.GetDimension(idx2) : SymbolicDimensionFactory.CreateKnown("dim_implicit_1", 1);

                SymbolicDimension outputDim = BroadcastingRule.Apply(dim1, dim2);
                outputDims.Insert(0, outputDim);
            }

            return new SymbolicShape(outputDims);
        }

        /// <summary>
        /// Gets the broadcast plan - a list of broadcasting rules for each dimension.
        /// </summary>
        /// <param name="shape1">The first shape.</param>
        /// <param name="shape2">The second shape.</param>
        /// <returns>A list of broadcasting rules.</returns>
        public List<BroadcastingRule> GetBroadcastPlan(SymbolicShape shape1, SymbolicShape shape2)
        {
            if (shape1 == null)
                throw new ArgumentNullException(nameof(shape1));
            if (shape2 == null)
                throw new ArgumentNullException(nameof(shape2));

            // Check cache
            var cacheKey = (shape1.GetHashCode(), shape2.GetHashCode());
            if (_broadcastPlanCache.TryGetValue(cacheKey, out var cachedPlan))
            {
                return cachedPlan;
            }

            if (!CanBroadcast(shape1, shape2))
            {
                throw new InvalidOperationException(
                    $"Cannot broadcast shapes {shape1} and {shape2}");
            }

            int rank1 = shape1.Rank;
            int rank2 = shape2.Rank;
            int maxRank = Math.Max(rank1, rank2);

            List<BroadcastingRule> plan = new List<BroadcastingRule>(maxRank);
            List<SymbolicDimension> outputDims = new List<SymbolicDimension>(maxRank);

            // Align shapes from right to left
            for (int i = 1; i <= maxRank; i++)
            {
                int idx1 = rank1 - i;
                int idx2 = rank2 - i;

                SymbolicDimension dim1 = idx1 >= 0 ? shape1.GetDimension(idx1) : SymbolicDimensionFactory.CreateKnown("dim_implicit_1", 1);
                SymbolicDimension dim2 = idx2 >= 0 ? shape2.GetDimension(idx2) : SymbolicDimensionFactory.CreateKnown("dim_implicit_1", 1);

                SymbolicDimension outputDim = BroadcastingRule.Apply(dim1, dim2);
                outputDims.Insert(0, outputDim);

                int dimensionIndex = maxRank - i;
                BroadcastingRule rule = new BroadcastingRule(dimensionIndex, dim1, dim2, outputDim);
                plan.Add(rule);
            }

            // Cache the plan
            _broadcastPlanCache[cacheKey] = plan;

            return plan;
        }

        /// <summary>
        /// Infers constraints that must be satisfied for broadcasting to work.
        /// </summary>
        /// <param name="shape1">The first shape.</param>
        /// <param name="shape2">The second shape.</param>
        /// <returns>A list of constraints that must be satisfied.</returns>
        public List<IShapeConstraint> InferBroadcastConstraints(SymbolicShape shape1, SymbolicShape shape2)
        {
            if (shape1 == null)
                throw new ArgumentNullException(nameof(shape1));
            if (shape2 == null)
                throw new ArgumentNullException(nameof(shape2));

            List<IShapeConstraint> constraints = new List<IShapeConstraint>();

            if (!CanBroadcast(shape1, shape2))
            {
                return constraints;
            }

            int rank1 = shape1.Rank;
            int rank2 = shape2.Rank;
            int maxRank = Math.Max(rank1, rank2);

            // Align shapes from right to left
            for (int i = 1; i <= maxRank; i++)
            {
                int idx1 = rank1 - i;
                int idx2 = rank2 - i;

                SymbolicDimension dim1 = idx1 >= 0 ? shape1.GetDimension(idx1) : SymbolicDimensionFactory.CreateKnown("dim_implicit_1", 1);
                SymbolicDimension dim2 = idx2 >= 0 ? shape2.GetDimension(idx2) : SymbolicDimensionFactory.CreateKnown("dim_implicit_1", 1);

                // If a dimension is symbolic and not 1, add constraint that it must be >= 1
                if (!dim1.Value.HasValue && (dim2.Value != 1 || dim2.Value == null))
                {
                    constraints.Add(new RangeConstraint(1, int.MaxValue));
                }

                if (!dim2.Value.HasValue && (dim1.Value != 1 || dim1.Value == null))
                {
                    constraints.Add(new RangeConstraint(1, int.MaxValue));
                }
            }

            return constraints;
        }

        /// <summary>
        /// Determines if two dimensions can be broadcast together.
        /// </summary>
        /// <param name="dim1">The first dimension.</param>
        /// <param name="dim2">The second dimension.</param>
        /// <returns>True if the dimensions are broadcastable; otherwise, false.</returns>
        private bool AreDimensionsBroadcastable(SymbolicDimension dim1, SymbolicDimension dim2)
        {
            // If dimensions are equal, they are broadcastable
            if (dim1.Value.HasValue && dim2.Value.HasValue && dim1.Value == dim2.Value)
            {
                return true;
            }

            // If one dimension is 1, they are broadcastable
            if (dim1.Value == 1 || dim2.Value == 1)
            {
                return true;
            }

            // If both are symbolic, they might be broadcastable (depends on constraints)
            if (!dim1.Value.HasValue && !dim2.Value.HasValue)
            {
                // For now, assume they could be equal or one could be 1
                return true;
            }

            // If one is symbolic and the other is not 1, the symbolic could be equal to the concrete
            if (!dim1.Value.HasValue || !dim2.Value.HasValue)
            {
                return true;
            }

            // Both are concrete, not equal, and neither is 1 - not broadcastable
            return false;
        }

        /// <summary>
        /// Clears the broadcast plan cache.
        /// </summary>
        public void ClearCache()
        {
            _broadcastPlanCache.Clear();
        }
    }
}
