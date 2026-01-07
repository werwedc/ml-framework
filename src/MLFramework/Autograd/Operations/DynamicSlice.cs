using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd.Operations
{
    /// <summary>
    /// Gradient function for dynamic slice operation with dynamic indices.
    /// Slices tensor with dynamic start positions and lengths.
    /// </summary>
    public sealed class DynamicSlice : IDynamicGradientFunction
    {
        private readonly int[] _startIndices;
        private readonly Shapes.SymbolicShape _sliceShape;

        /// <summary>
        /// Initializes a new instance of the DynamicSlice class.
        /// </summary>
        /// <param name="startIndices">The start indices for the slice.</param>
        /// <param name="sliceShape">The shape of the slice.</param>
        public DynamicSlice(int[] startIndices, Shapes.SymbolicShape sliceShape)
        {
            if (startIndices == null || startIndices.Length == 0)
            {
                throw new ArgumentException("Start indices must be non-empty", nameof(startIndices));
            }

            if (sliceShape == null)
                throw new ArgumentNullException(nameof(sliceShape));

            _startIndices = startIndices;
            _sliceShape = sliceShape;
        }

        /// <summary>
        /// Gets the start indices.
        /// </summary>
        public int[] StartIndices => _startIndices;

        /// <summary>
        /// Gets the slice shape.
        /// </summary>
        public Shapes.SymbolicShape SliceShape => _sliceShape;

        /// <summary>
        /// Gets the name of this operation.
        /// </summary>
        public string OperationName => "DynamicSlice";

        /// <summary>
        /// Computes the output shape (slice shape).
        /// </summary>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <returns>The shape of the output tensor.</returns>
        public Shapes.SymbolicShape GetOutputShape(System.Collections.Generic.List<Shapes.SymbolicShape> inputShapes)
        {
            if (inputShapes == null || inputShapes.Count != 1)
            {
                throw new ArgumentException("DynamicSlice requires exactly 1 input shape");
            }

            return _sliceShape;
        }

        /// <summary>
        /// Computes the output shapes for all outputs.
        /// </summary>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <returns>A list of output shapes (single element for DynamicSlice).</returns>
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

            // Gradient shape should match slice shape
            if (!gradientShape.Equals(_sliceShape))
            {
                throw new ArgumentException(
                    $"Gradient shape {gradientShape} does not match slice shape {_sliceShape}");
            }
        }

        /// <summary>
        /// Computes the gradient for the dynamic slice operation.
        /// </summary>
        /// <param name="gradOutput">The gradient from downstream operations.</param>
        /// <param name="inputs">The input tensors to the original operation.</param>
        /// <param name="context">The operation context containing saved intermediate values.</param>
        /// <returns>An array of gradients with unslicing applied.</returns>
        public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
        {
            if (gradOutput == null)
                throw new ArgumentNullException(nameof(gradOutput));

            if (inputs == null || inputs.Length != 1)
            {
                throw new ArgumentException("DynamicSlice requires exactly 1 input");
            }

            var inputTensor = inputs[0];

            // Gradient is placed back into the original tensor shape
            var dInput = UnsliceGradient(gradOutput, inputTensor.Shape);

            return new Tensor[] { dInput };
        }

        /// <summary>
        /// Unslices the gradient tensor, placing it back into the original shape.
        /// </summary>
        /// <param name="gradient">The gradient tensor to unslice.</param>
        /// <param name="inputShape">The original input tensor shape.</param>
        /// <returns>The unsliced gradient tensor.</returns>
        private Tensor UnsliceGradient(Tensor gradient, int[] inputShape)
        {
            // In practice, this would use a tensor library to unslice the gradient
            // The gradient is placed into the original shape at the slice position
            // For now, we return gradient as a placeholder
            // TODO: Implement proper unsliding with tensor library

            return gradient;
        }
    }
}
