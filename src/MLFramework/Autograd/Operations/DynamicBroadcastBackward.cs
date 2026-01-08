using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd.Operations
{
    /// <summary>
    /// Gradient function for broadcast operations with dynamic shapes.
    /// Sums gradient over broadcasted dimensions and handles reduction with symbolic sizes.
    /// </summary>
    public sealed class DynamicBroadcastBackward : IDynamicGradientFunction
    {
        private readonly Shapes.SymbolicShape _inputShape;
        private readonly Shapes.SymbolicShape _outputShape;
        private readonly int[] _broadcastAxes;

        /// <summary>
        /// Initializes a new instance of the DynamicBroadcastBackward class.
        /// </summary>
        /// <param name="inputShape">The shape of the input before broadcasting.</param>
        /// <param name="outputShape">The shape after broadcasting.</param>
        /// <param name="broadcastAxes">The axes over which broadcasting occurred.</param>
        public DynamicBroadcastBackward(
            Shapes.SymbolicShape inputShape,
            Shapes.SymbolicShape outputShape,
            int[]? broadcastAxes = null)
        {
            _inputShape = inputShape ?? throw new ArgumentNullException(nameof(inputShape));
            _outputShape = outputShape ?? throw new ArgumentNullException(nameof(outputShape));
            _broadcastAxes = broadcastAxes ?? Array.Empty<int>();
        }

        /// <summary>
        /// Gets the input shape before broadcasting.
        /// </summary>
        public Shapes.SymbolicShape InputShape => _inputShape;

        /// <summary>
        /// Gets the output shape after broadcasting.
        /// </summary>
        public Shapes.SymbolicShape OutputShape => _outputShape;

        /// <summary>
        /// Gets the axes over which broadcasting occurred.
        /// </summary>
        public int[] BroadcastAxes => _broadcastAxes;

        /// <summary>
        /// Gets the name of this operation.
        /// </summary>
        public string OperationName => "Broadcast";

        /// <summary>
        /// Computes the output shape (same as broadcasted shape).
        /// </summary>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <returns>The shape of the output tensor.</returns>
        public Shapes.SymbolicShape GetOutputShape(System.Collections.Generic.List<Shapes.SymbolicShape> inputShapes)
        {
            if (inputShapes == null || inputShapes.Count != 1)
            {
                throw new ArgumentException("Broadcast requires exactly 1 input shape");
            }

            return _outputShape;
        }

        /// <summary>
        /// Computes the output shapes for all outputs.
        /// </summary>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <returns>A list of output shapes (single element for Broadcast).</returns>
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

            // Gradient shape should match output shape (not input shape)
            if (!gradientShape.Equals(_outputShape))
            {
                throw new ArgumentException(
                    $"Gradient shape {gradientShape} does not match output shape {_outputShape}");
            }
        }

        /// <summary>
        /// Computes the gradient for the broadcast operation.
        /// </summary>
        /// <param name="gradOutput">The gradient from downstream operations.</param>
        /// <param name="inputs">The input tensors to the original operation.</param>
        /// <param name="context">The operation context containing saved intermediate values.</param>
        /// <returns>An array of gradients summed over broadcasted dimensions.</returns>
        public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
        {
            if (gradOutput == null)
                throw new ArgumentNullException(nameof(gradOutput));

            if (inputs == null || inputs.Length != 1)
            {
                throw new ArgumentException("Broadcast requires exactly 1 input");
            }

            // Gradient is summed over broadcasted dimensions
            var dInput = ReduceGradientOverBroadcastedAxes(gradOutput);

            return new Tensor[] { dInput };
        }

        /// <summary>
        /// Reduces the gradient over the broadcasted axes.
        /// </summary>
        /// <param name="gradOutput">The gradient tensor to reduce.</param>
        /// <returns>The reduced gradient tensor.</returns>
        private Tensor ReduceGradientOverBroadcastedAxes(Tensor gradOutput)
        {
            if (_broadcastAxes.Length == 0)
            {
                // No broadcasting occurred, gradient flows directly
                return gradOutput;
            }

            // Sum gradient over broadcasted axes
            // Dimensions that were broadcast from size 1 need to be summed
            // Dimensions that were not broadcast (matching sizes) are kept as-is
            Tensor reduced = gradOutput;

            // Determine which axes to reduce over
            var axesToReduce = GetAxesToReduce();

            if (axesToReduce.Length > 0)
            {
                // Sum over the broadcasted axes
                reduced = SumOverAxes(reduced, axesToReduce);
            }

            // Reshape to match input shape
            reduced = ReshapeToInputShape(reduced);

            return reduced;
        }

        /// <summary>
        /// Determines which axes need to be reduced based on input and output shapes.
        /// </summary>
        /// <returns>An array of axes to reduce.</returns>
        private int[] GetAxesToReduce()
        {
            var axesToReduce = new List<int>();

            // Compare input and output shapes to find broadcasted dimensions
            int maxRank = Math.Max(_inputShape.Rank, _outputShape.Rank);

            for (int i = 0; i < maxRank; i++)
            {
                var inputDim = i < _inputShape.Rank
                    ? _inputShape.GetDimension(-(i + 1))
                    : new Shapes.SymbolicDimension($"_dummy_{i}", 1);

                var outputDim = i < _outputShape.Rank
                    ? _outputShape.GetDimension(-(i + 1))
                    : new Shapes.SymbolicDimension($"_dummy_{i}", 1);

                // If input dimension was 1 and output is larger, we need to reduce
                if (inputDim.IsKnown() && inputDim.Value == 1 &&
                    (!outputDim.IsKnown() || outputDim.Value > 1))
                {
                    axesToReduce.Add(maxRank - i - 1);
                }
            }

            return axesToReduce.ToArray();
        }

        /// <summary>
        /// Sums a tensor over the specified axes.
        /// </summary>
        /// <param name="tensor">The tensor to sum.</param>
        /// <param name="axes">The axes to sum over.</param>
        /// <returns>The summed tensor.</returns>
        private static Tensor SumOverAxes(Tensor tensor, int[] axes)
        {
            // In practice, this would use a tensor library to sum over axes
            // For now, we return tensor as a placeholder
            // TODO: Implement proper sum operation with tensor library

            return tensor;
        }

        /// <summary>
        /// Reshapes the reduced gradient to match the input shape.
        /// </summary>
        /// <param name="tensor">The tensor to reshape.</param>
        /// <returns>The reshaped tensor.</returns>
        private Tensor ReshapeToInputShape(Tensor tensor)
        {
            // In practice, this would use a tensor library to reshape
            // For now, we return tensor as a placeholder
            // TODO: Implement proper reshape operation with tensor library

            return tensor;
        }
    }
}
