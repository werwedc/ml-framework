using MLFramework.Shapes;
using MLFramework.Shapes.Inference;

namespace MLFramework.Shapes.Inference.Rules
{
    /// <summary>
    /// Shape inference rule for reshape operations.
    /// </summary>
    public class ReshapeRule : ShapeInferenceRuleBase
    {
        /// <summary>
        /// Gets the supported operations.
        /// </summary>
        protected override string[] SupportedOperations => new[] { "Reshape", "View", "Flatten" };

        /// <summary>
        /// Gets the expected input count.
        /// </summary>
        protected override int GetExpectedInputCount(string opName)
        {
            return 2; // Input tensor and target shape
        }

        /// <summary>
        /// Infers the output shape for reshape.
        /// Inputs: [input_tensor_shape, target_shape]
        /// Output: [target_shape_with_inferred_dims]
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <param name="inputs">The input shapes.</param>
        /// <returns>The inferred output shape.</returns>
        protected override List<SymbolicShape> InferOutputShapes(string opName, IReadOnlyList<SymbolicShape> inputs)
        {
            var inputShape = inputs[0]; // Shape of the tensor to reshape
            var targetShape = inputs[1]; // Target shape

            // Validate input rank (should be a shape itself, so rank could be any)
            // For reshape, the second input is typically a concrete 1D tensor representing the target shape
            // But for shape inference, we receive it as a SymbolicShape

            // Calculate total elements in input
            if (!inputShape.IsFullyKnown())
            {
                // If input shape is not fully known, we can't validate element count
                // Return the target shape as-is
                return new List<SymbolicShape> { targetShape.Clone() };
            }

            var inputConcreteShape = inputShape.ToConcrete();
            long inputElements = CalculateTotalElements(inputConcreteShape);

            // Handle -1 dimension in target shape
            var targetDims = new List<SymbolicDimension>();
            int negOneIndex = -1;
            long knownElements = 1;

            for (int i = 0; i < targetShape.Rank; i++)
            {
                var dim = targetShape.GetDimension(i);

                if (dim.IsKnown() && dim.Value!.Value == -1)
                {
                    // Found -1 dimension
                    if (negOneIndex != -1)
                    {
                        throw new ArgumentException(
                            $"Target shape can have at most one -1 dimension, found multiple for '{opName}'");
                    }
                    negOneIndex = i;
                    targetDims.Add(dim);
                }
                else if (dim.IsKnown())
                {
                    knownElements *= dim.Value!.Value;
                    targetDims.Add(dim);
                }
                else
                {
                    // Symbolic dimension - assume it's valid
                    targetDims.Add(dim);
                }
            }

            // Infer -1 dimension if present
            if (negOneIndex != -1)
            {
                if (inputElements % knownElements != 0)
                {
                    throw new ArgumentException(
                        $"Cannot reshape {inputShape} to {targetShape}: " +
                        $"total elements ({inputElements}) is not divisible by known elements ({knownElements})");
                }

                int inferredValue = (int)(inputElements / knownElements);
                targetDims[negOneIndex] = SymbolicDimensionFactory.Create("inferred_dim", inferredValue);
            }
            else
            {
                // Validate total elements match
                if (targetShape.IsFullyKnown())
                {
                    var targetConcreteShape = targetShape.ToConcrete();
                    long targetElements = CalculateTotalElements(targetConcreteShape);

                    if (inputElements != targetElements)
                    {
                        throw new ArgumentException(
                            $"Cannot reshape {inputShape} to {targetShape}: " +
                            $"element count mismatch ({inputElements} vs {targetElements})");
                    }
                }
            }

            return new List<SymbolicShape> { new SymbolicShape(targetDims) };
        }

        /// <summary>
        /// Calculates the total number of elements in a shape.
        /// </summary>
        private long CalculateTotalElements(int[] shape)
        {
            long total = 1;
            foreach (int dim in shape)
            {
                total *= dim;
            }
            return total;
        }

        /// <summary>
        /// Flattens a tensor to 1D.
        /// </summary>
        /// <param name="inputShape">The input shape.</param>
        /// <returns>The flattened shape [N].</returns>
        public SymbolicShape Flatten(SymbolicShape inputShape)
        {
            if (inputShape == null)
                throw new ArgumentNullException(nameof(inputShape));

            // If input shape is fully known, calculate total elements
            if (inputShape.IsFullyKnown())
            {
                var concreteShape = inputShape.ToConcrete();
                long totalElements = CalculateTotalElements(concreteShape);
                return new SymbolicShape(SymbolicDimensionFactory.Create("N", (int)totalElements));
            }

                // Otherwise, return a symbolic dimension representing the flattened size
                return new SymbolicShape(SymbolicDimensionFactory.Create("N"));
        }
    }
}
