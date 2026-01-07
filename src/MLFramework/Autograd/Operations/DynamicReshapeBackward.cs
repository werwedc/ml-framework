using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd.Operations
{
    /// <summary>
    /// Gradient function for reshape operation with dynamic shapes.
    /// Reshapes gradient to original shape, preserving symbolic dimensions.
    /// </summary>
    public sealed class DynamicReshapeBackward : IDynamicGradientFunction
    {
        private readonly Shapes.SymbolicShape _originalShape;

        /// <summary>
        /// Initializes a new instance of the DynamicReshapeBackward class.
        /// </summary>
        /// <param name="originalShape">The original shape before reshape.</param>
        public DynamicReshapeBackward(Shapes.SymbolicShape originalShape)
        {
            _originalShape = originalShape ?? throw new ArgumentNullException(nameof(originalShape));
        }

        /// <summary>
        /// Gets the original shape.
        /// </summary>
        public Shapes.SymbolicShape OriginalShape => _originalShape;

        /// <summary>
        /// Gets the name of this operation.
        /// </summary>
        public string OperationName => "Reshape";

        /// <summary>
        /// Computes the output shape (same as the target shape in the forward pass).
        /// </summary>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <returns>The shape of the output tensor.</returns>
        public Shapes.SymbolicShape GetOutputShape(System.Collections.Generic.List<Shapes.SymbolicShape> inputShapes)
        {
            if (inputShapes == null || inputShapes.Count != 1)
            {
                throw new ArgumentException("Reshape requires exactly 1 input shape");
            }

            // For backward, we need to return the original shape
            return _originalShape;
        }

        /// <summary>
        /// Computes the output shapes for all outputs.
        /// </summary>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <returns>A list of output shapes (single element for Reshape).</returns>
        public System.Collections.Generic.List<Shapes.SymbolicShape> GetOutputShapes(
            System.Collections.Generic.List<Shapes.SymbolicShape> inputShapes)
        {
            return new System.Collections.Generic.List<Shapes.SymbolicShape>
            {
                GetOutputShape(inputShapes)
            };
        }

        /// <summary>
        /// Validates that a gradient shape matches the expected input shape.
        /// </summary>
        /// <param name="gradientShape">The shape of the gradient tensor.</param>
        /// <exception cref="ArgumentException">Thrown when the gradient shape is invalid.</exception>
        public void ValidateGradientShape(Shapes.SymbolicShape gradientShape)
        {
            if (gradientShape == null)
            {
                throw new ArgumentNullException(nameof(gradientShape));
            }

            // Validate that gradient shape is compatible with original shape
            // They should have the same total size
            if (_originalShape.IsFullyKnown() && gradientShape.IsFullyKnown())
            {
                long originalSize = _originalShape.ToConcrete()
                    .Aggregate(1L, (acc, dim) => acc * dim);
                long gradientSize = gradientShape.ToConcrete()
                    .Aggregate(1L, (acc, dim) => acc * dim);

                if (originalSize != gradientSize)
                {
                    throw new ArgumentException(
                        $"Gradient shape {gradientShape} is not compatible with original shape {_originalShape}. " +
                        $"Sizes: {originalSize} vs {gradientSize}");
                }
            }
        }

        /// <summary>
        /// Computes the gradient for the reshape operation.
        /// </summary>
        /// <param name="gradOutput">The gradient from downstream operations.</param>
        /// <param name="inputs">The input tensors to the original operation.</param>
        /// <param name="context">The operation context containing saved intermediate values.</param>
        /// <returns>An array of gradients reshaped to the original shape.</returns>
        public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
        {
            if (gradOutput == null)
                throw new ArgumentNullException(nameof(gradOutput));

            if (inputs == null || inputs.Length != 1)
            {
                throw new ArgumentException("Reshape requires exactly 1 input");
            }

            // Gradient flows directly - just need to reshape
            // Reshape gradOutput to match original input shape
            var dInput = ReshapeGradient(gradOutput);

            return new Tensor[] { dInput };
        }

        /// <summary>
        /// Reshapes the gradient tensor to the original shape.
        /// </summary>
        /// <param name="gradOutput">The gradient tensor to reshape.</param>
        /// <returns>The reshaped gradient tensor.</returns>
        private Tensor ReshapeGradient(Tensor gradOutput)
        {
            // In practice, this would use a tensor library to reshape
            // For now, we return gradOutput as a placeholder
            // TODO: Implement proper reshape operation with tensor library

            return gradOutput;
        }
    }
}
