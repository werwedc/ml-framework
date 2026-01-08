using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd.Operations
{
    /// <summary>
    /// Gradient function for dynamic gather operation with dynamic indices.
    /// Gathers elements at dynamic indices and handles gradient scattering.
    /// </summary>
    public sealed class DynamicGather : IDynamicGradientFunction
    {
        private readonly int _axis;

        /// <summary>
        /// Initializes a new instance of the DynamicGather class.
        /// </summary>
        /// <param name="axis">The axis along which to gather.</param>
        public DynamicGather(int axis)
        {
            _axis = axis;
        }

        /// <summary>
        /// Gets the axis along which gathering occurs.
        /// </summary>
        public int Axis => _axis;

        /// <summary>
        /// Gets the name of this operation.
        /// </summary>
        public string OperationName => "DynamicGather";

        /// <summary>
        /// Computes the output shape for the gather operation.
        /// </summary>
        /// <param name="inputShapes">The shapes of the input tensors [data, indices].</param>
        /// <returns>The shape of the output tensor.</returns>
        public Shapes.SymbolicShape GetOutputShape(System.Collections.Generic.List<Shapes.SymbolicShape> inputShapes)
        {
            if (inputShapes == null || inputShapes.Count != 2)
            {
                throw new ArgumentException("DynamicGather requires exactly 2 input shapes (data and indices)");
            }

            var dataShape = inputShapes[0];
            var indicesShape = inputShapes[1];

            // Output shape is like indices shape, but with the gathered dimension from data
            // If axis=0:
            //   Data: [D0, D1, ..., Dn]
            //   Indices: [I0, I1, ..., Im]
            //   Output: [I0, I1, ..., Im, D1, ..., Dn]

            var outputDims = new List<Shapes.SymbolicDimension>();

            // Add all dimensions from indices
            for (int i = 0; i < indicesShape.Rank; i++)
            {
                outputDims.Add(indicesShape.GetDimension(i));
            }

            // Add all dimensions from data except the axis dimension
            for (int i = 0; i < dataShape.Rank; i++)
            {
                if (i != _axis)
                {
                    outputDims.Add(dataShape.GetDimension(i));
                }
            }

            return new Shapes.SymbolicShape(outputDims);
        }

        /// <summary>
        /// Computes the output shapes for all outputs.
        /// </summary>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <returns>A list of output shapes (single element for DynamicGather).</returns>
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

            // Basic validation - gather typically outputs at least 1 dimension
            if (gradientShape.Rank < 1)
            {
                throw new ArgumentException(
                    $"Gradient shape must have rank >= 1, got {gradientShape.Rank}");
            }
        }

        /// <summary>
        /// Computes the gradient for the dynamic gather operation.
        /// </summary>
        /// <param name="gradOutput">The gradient from downstream operations.</param>
        /// <param name="inputs">The input tensors [data, indices].</param>
        /// <param name="context">The operation context containing saved intermediate values.</param>
        /// <returns>An array of gradients [dData, dIndices].</returns>
        public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
        {
            if (gradOutput == null)
                throw new ArgumentNullException(nameof(gradOutput));

            if (inputs == null || inputs.Length != 2)
            {
                throw new ArgumentException("DynamicGather requires exactly 2 inputs (data and indices)");
            }

            var dataTensor = inputs[0];
            var indicesTensor = inputs[1];

            // Gradient for data: scatter the gradient back to the original positions
            var dData = ScatterGradientBack(gradOutput, dataTensor.Shape, indicesTensor);

            // Gradient for indices is zero (indices are discrete)
            var dIndices = ZeroGradient(indicesTensor.Shape);

            return new Tensor[] { dData, dIndices };
        }

        /// <summary>
        /// Scatters the gradient back to the original data shape.
        /// </summary>
        /// <param name="gradient">The gradient tensor to scatter.</param>
        /// <param name="dataShape">The shape of the original data tensor.</param>
        /// <param name="indices">The indices tensor from the forward pass.</param>
        /// <returns>The scattered gradient tensor.</returns>
        private Tensor ScatterGradientBack(Tensor gradient, int[] dataShape, Tensor indices)
        {
            // In practice, this would use a tensor library to scatter the gradient
            // The gradient is placed back at the indices from the forward pass
            // For now, we return gradient as a placeholder
            // TODO: Implement proper scatter operation with tensor library

            return gradient;
        }

        /// <summary>
        /// Creates a zero gradient tensor with the specified shape.
        /// </summary>
        /// <param name="shape">The shape of the tensor.</param>
        /// <returns>A zero gradient tensor.</returns>
        private static Tensor ZeroGradient(int[] shape)
        {
            // In practice, this would use a tensor library to create a zero tensor
            // For now, we return a placeholder
            // TODO: Implement proper zero tensor creation with tensor library

            return null!;
        }
    }
}
