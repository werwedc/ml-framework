using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd.Operations
{
    /// <summary>
    /// Gradient function for dynamic pad operation with variable sizes.
    /// Pads tensor to variable sizes and handles gradient unpadding.
    /// </summary>
    public sealed class DynamicPad : IDynamicGradientFunction
    {
        private readonly int[] _padding;
        private readonly Shapes.SymbolicShape _targetShape;

        /// <summary>
        /// Initializes a new instance of the DynamicPad class.
        /// </summary>
        /// <param name="padding">The padding amounts for each dimension.</param>
        /// <param name="targetShape">The target shape after padding.</param>
        public DynamicPad(int[] padding, Shapes.SymbolicShape targetShape)
        {
            if (padding == null || padding.Length == 0)
            {
                throw new ArgumentException("Padding must be non-empty", nameof(padding));
            }

            if (targetShape == null)
                throw new ArgumentNullException(nameof(targetShape));

            _padding = padding;
            _targetShape = targetShape;
        }

        /// <summary>
        /// Gets the padding amounts.
        /// </summary>
        public int[] Padding => _padding;

        /// <summary>
        /// Gets the target shape after padding.
        /// </summary>
        public Shapes.SymbolicShape TargetShape => _targetShape;

        /// <summary>
        /// Gets the name of this operation.
        /// </summary>
        public string OperationName => "DynamicPad";

        /// <summary>
        /// Computes the output shape (target shape).
        /// </summary>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <returns>The shape of the output tensor.</returns>
        public Shapes.SymbolicShape GetOutputShape(System.Collections.Generic.List<Shapes.SymbolicShape> inputShapes)
        {
            if (inputShapes == null || inputShapes.Count != 1)
            {
                throw new ArgumentException("DynamicPad requires exactly 1 input shape");
            }

            return _targetShape;
        }

        /// <summary>
        /// Computes the output shapes for all outputs.
        /// </summary>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <returns>A list of output shapes (single element for DynamicPad).</returns>
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

            // Gradient shape should match target shape (padded shape)
            if (!gradientShape.Equals(_targetShape))
            {
                throw new ArgumentException(
                    $"Gradient shape {gradientShape} does not match target shape {_targetShape}");
            }
        }

        /// <summary>
        /// Computes the gradient for the dynamic pad operation.
        /// </summary>
        /// <param name="gradOutput">The gradient from downstream operations.</param>
        /// <param name="inputs">The input tensors to the original operation.</param>
        /// <param name="context">The operation context containing saved intermediate values.</param>
        /// <returns>An array of gradients with padding removed.</returns>
        public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
        {
            if (gradOutput == null)
                throw new ArgumentNullException(nameof(gradOutput));

            if (inputs == null || inputs.Length != 1)
            {
                throw new ArgumentException("DynamicPad requires exactly 1 input");
            }

            // Gradient is unpadded (inverse of forward operation)
            var dInput = RemovePaddingFromGradient(gradOutput);

            return new Tensor[] { dInput };
        }

        /// <summary>
        /// Removes the padding from the gradient tensor.
        /// </summary>
        /// <param name="gradient">The gradient tensor to unpad.</param>
        /// <returns>The unpadded gradient tensor.</returns>
        private Tensor RemovePaddingFromGradient(Tensor gradient)
        {
            // In practice, this would use a tensor library to remove padding
            // For now, we return gradient as a placeholder
            // TODO: Implement proper padding removal with tensor library

            return gradient;
        }
    }
}
