using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd.Operations
{
    /// <summary>
    /// Gradient function for dynamic mask operation with variable sequence lengths.
    /// Masks out values beyond the valid sequence length.
    /// </summary>
    public sealed class DynamicMask : IDynamicGradientFunction
    {
        private readonly Shapes.SymbolicDimension _sequenceLength;

        /// <summary>
        /// Initializes a new instance of the DynamicMask class.
        /// </summary>
        /// <param name="sequenceLength">The sequence length dimension.</param>
        public DynamicMask(Shapes.SymbolicDimension sequenceLength)
        {
            _sequenceLength = sequenceLength ?? throw new ArgumentNullException(nameof(sequenceLength));
        }

        /// <summary>
        /// Gets the sequence length dimension.
        /// </summary>
        public Shapes.SymbolicDimension SequenceLength => _sequenceLength;

        /// <summary>
        /// Gets the name of this operation.
        /// </summary>
        public string OperationName => "DynamicMask";

        /// <summary>
        /// Computes the output shape (same as input shape).
        /// </summary>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <returns>The shape of the output tensor.</returns>
        public Shapes.SymbolicShape GetOutputShape(System.Collections.Generic.List<Shapes.SymbolicShape> inputShapes)
        {
            if (inputShapes == null || inputShapes.Count != 1)
            {
                throw new ArgumentException("DynamicMask requires exactly 1 input shape");
            }

            // Mask operation doesn't change shape
            return inputShapes[0];
        }

        /// <summary>
        /// Computes the output shapes for all outputs.
        /// </summary>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <returns>A list of output shapes (single element for DynamicMask).</returns>
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

            // Basic validation - shape should have at least 2 dimensions for batch and sequence
            if (gradientShape.Rank < 2)
            {
                throw new ArgumentException(
                    $"Gradient shape must have rank >= 2, got {gradientShape.Rank}");
            }
        }

        /// <summary>
        /// Computes the gradient for the dynamic mask operation.
        /// </summary>
        /// <param name="gradOutput">The gradient from downstream operations.</param>
        /// <param name="inputs">The input tensors to the original operation.</param>
        /// <param name="context">The operation context containing saved intermediate values.</param>
        /// <returns>An array of gradients with masking applied.</returns>
        public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
        {
            if (gradOutput == null)
                throw new ArgumentNullException(nameof(gradOutput));

            if (inputs == null || inputs.Length != 1)
            {
                throw new ArgumentException("DynamicMask requires exactly 1 input");
            }

            // Gradient is also masked - same operation as forward
            var dInput = ApplyMaskToGradient(gradOutput);

            return new Tensor[] { dInput };
        }

        /// <summary>
        /// Applies the mask to the gradient tensor.
        /// </summary>
        /// <param name="gradient">The gradient tensor to mask.</param>
        /// <returns>The masked gradient tensor.</returns>
        private Tensor ApplyMaskToGradient(Tensor gradient)
        {
            // In practice, this would use a tensor library to apply the mask
            // For now, we return gradient as a placeholder
            // TODO: Implement proper mask application with tensor library

            return gradient;
        }
    }
}
