using MLFramework.Shapes;
using MLFramework.Shapes.Inference;

namespace MLFramework.Shapes.Inference.Rules
{
    /// <summary>
    /// Shape inference rule for transpose operations.
    /// </summary>
    public class TransposeRule : ShapeInferenceRuleBase
    {
        /// <summary>
        /// Gets the supported operations.
        /// </summary>
        protected override string[] SupportedOperations => new[] { "Transpose", "Transpose2d" };

        /// <summary>
        /// Gets the expected input count.
        /// </summary>
        protected override int GetExpectedInputCount(string opName)
        {
            return 1;
        }

        /// <summary>
        /// Infers the output shape for transpose.
        /// Note: This implementation assumes the permutation is provided separately
        /// (e.g., as an attribute). For this simplified version, we'll assume
        /// the user provides permutation along with the input shapes.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <param name="inputs">The input shapes.</param>
        /// <returns>The inferred output shape.</returns>
        protected override List<SymbolicShape> InferOutputShapes(string opName, IReadOnlyList<SymbolicShape> inputs)
        {
            // For a proper implementation, we'd need to access the permutation attribute
            // This is a simplified version that assumes the input is a 2D matrix
            // and performs a simple transpose (swap dimensions 0 and 1)

            var inputShape = inputs[0];

            // For 2D input: swap dimensions
            if (inputShape.Rank == 2)
            {
                var dim0 = inputShape.GetDimension(0);
                var dim1 = inputShape.GetDimension(1);
                return new List<SymbolicShape> { new SymbolicShape(dim1, dim0) };
            }

            // For nD input, we need a permutation
            // Since we don't have access to attributes in this simplified API,
            // we'll throw an exception suggesting the use of a more advanced API
            throw new NotImplementedException(
                $"Transpose for tensors with rank {inputShape.Rank} requires a permutation. " +
                $"Please use the advanced API that supports passing operation attributes.");
        }

        /// <summary>
        /// Infers the output shape for transpose with a specified permutation.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <param name="inputShape">The input shape.</param>
        /// <param name="permutation">The permutation of axes.</param>
        /// <returns>The inferred output shape.</returns>
        public SymbolicShape InferWithPermutation(string opName, SymbolicShape inputShape, int[] permutation)
        {
            if (inputShape == null)
                throw new ArgumentNullException(nameof(inputShape));

            if (permutation == null)
                throw new ArgumentNullException(nameof(permutation));

            if (permutation.Length != inputShape.Rank)
            {
                throw new ArgumentException(
                    $"Permutation length ({permutation.Length}) must match input rank ({inputShape.Rank})");
            }

            // Validate permutation is valid (contains all indices from 0 to rank-1 exactly once)
            var sortedPerm = permutation.OrderBy(x => x).ToArray();
            for (int i = 0; i < sortedPerm.Length; i++)
            {
                if (sortedPerm[i] != i)
                {
                    throw new ArgumentException($"Invalid permutation: {string.Join(", ", permutation)}");
                }
            }

            // Apply permutation to dimensions
            var outputDims = new SymbolicDimension[permutation.Length];
            for (int i = 0; i < permutation.Length; i++)
            {
                outputDims[i] = inputShape.GetDimension(permutation[i]);
            }

            return new SymbolicShape(outputDims);
        }
    }
}
