using MLFramework.Shapes;

namespace MLFramework.Shapes.Inference
{
    /// <summary>
    /// Abstract base class for shape inference rules with common validation logic.
    /// </summary>
    public abstract class ShapeInferenceRuleBase : IShapeInferenceRule
    {
        /// <summary>
        /// Gets the operation names this rule supports.
        /// </summary>
        protected abstract string[] SupportedOperations { get; }

        /// <summary>
        /// Determines whether this rule can infer the output shapes for the given operation and input shapes.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <param name="inputs">The input shapes.</param>
        /// <returns>True if this rule can infer the output shapes; otherwise, false.</returns>
        public virtual bool CanInfer(string opName, IReadOnlyList<SymbolicShape> inputs)
        {
            if (string.IsNullOrEmpty(opName))
                return false;

            if (!SupportedOperations.Contains(opName))
                return false;

            // Check input count
            int expectedInputCount = GetExpectedInputCount(opName);
            if (expectedInputCount >= 0 && inputs.Count != expectedInputCount)
                return false;

            return true;
        }

        /// <summary>
        /// Infers the output shapes for the given operation and input shapes.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <param name="inputs">The input shapes.</param>
        /// <returns>The inferred output shapes.</returns>
        public List<SymbolicShape> Infer(string opName, IReadOnlyList<SymbolicShape> inputs)
        {
            if (!CanInfer(opName, inputs))
            {
                throw new ArgumentException(
                    $"Cannot infer shape for operation '{opName}' with {inputs.Count} input(s)");
            }

            ValidateInputs(opName, inputs);
            return InferOutputShapes(opName, inputs);
        }

        /// <summary>
        /// Gets the expected number of inputs for the given operation.
        /// Returns -1 if the operation accepts variable number of inputs.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <returns>The expected number of inputs, or -1 for variable count.</returns>
        protected virtual int GetExpectedInputCount(string opName)
        {
            return -1; // Default: variable count
        }

        /// <summary>
        /// Validates that the input shapes are valid for the operation.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <param name="inputs">The input shapes to validate.</param>
        protected virtual void ValidateInputs(string opName, IReadOnlyList<SymbolicShape> inputs)
        {
            // Default implementation: check for null inputs
            for (int i = 0; i < inputs.Count; i++)
            {
                if (inputs[i] == null)
                {
                    throw new ArgumentException(
                        $"Input shape at index {i} is null for operation '{opName}'");
                }
            }
        }

        /// <summary>
        /// Infers the output shapes for the given operation and input shapes.
        /// Must be implemented by derived classes.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <param name="inputs">The input shapes.</param>
        /// <returns>The inferred output shapes.</returns>
        protected abstract List<SymbolicShape> InferOutputShapes(string opName, IReadOnlyList<SymbolicShape> inputs);

        /// <summary>
        /// Validates that a shape has the expected rank.
        /// </summary>
        /// <param name="shape">The shape to validate.</param>
        /// <param name="expectedRank">The expected rank.</param>
        /// <param name="inputIndex">The index of the input (for error messages).</param>
        /// <param name="opName">The name of the operation (for error messages).</param>
        protected void ValidateRank(SymbolicShape shape, int expectedRank, int inputIndex, string opName)
        {
            if (shape.Rank != expectedRank)
            {
                throw new ArgumentException(
                    $"Input {inputIndex} for operation '{opName}' must have rank {expectedRank}, " +
                    $"but has rank {shape.Rank}");
            }
        }

        /// <summary>
        /// Validates that two shapes have compatible ranks for broadcasting.
        /// </summary>
        /// <param name="shapeA">The first shape.</param>
        /// <param name="shapeB">The second shape.</param>
        /// <param name="opName">The name of the operation (for error messages).</param>
        protected void ValidateBroadcastRanks(SymbolicShape shapeA, SymbolicShape shapeB, string opName)
        {
            // Shapes are compatible for broadcasting if they have the same rank,
            // or one has fewer dimensions (prepended with 1's)
            // This is just a preliminary check - full validation happens in broadcasting logic
            return;
        }
    }
}
