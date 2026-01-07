using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd.Operations
{
    /// <summary>
    /// Gradient function for 2D convolution with dynamic shapes.
    /// Computes gradients with dynamic spatial dimensions and handles padding/stride with unknown sizes.
    /// </summary>
    public sealed class DynamicConv2DBackward : IDynamicGradientFunction
    {
        private readonly int[] _kernelSize;
        private readonly int[] _stride;
        private readonly int[] _padding;
        private readonly int[] _dilation;

        /// <summary>
        /// Initializes a new instance of the DynamicConv2DBackward class.
        /// </summary>
        /// <param name="kernelSize">The kernel size [height, width].</param>
        /// <param name="stride">The stride [height, width]. Defaults to [1, 1].</param>
        /// <param name="padding">The padding [height, width]. Defaults to [0, 0].</param>
        /// <param name="dilation">The dilation [height, width]. Defaults to [1, 1].</param>
        public DynamicConv2DBackward(
            int[] kernelSize,
            int[]? stride = null,
            int[]? padding = null,
            int[]? dilation = null)
        {
            if (kernelSize == null || kernelSize.Length != 2)
            {
                throw new ArgumentException("Kernel size must have exactly 2 elements", nameof(kernelSize));
            }

            _kernelSize = kernelSize;
            _stride = stride ?? new[] { 1, 1 };
            _padding = padding ?? new[] { 0, 0 };
            _dilation = dilation ?? new[] { 1, 1 };
        }

        /// <summary>
        /// Gets the name of this operation.
        /// </summary>
        public string OperationName => "Conv2D";

        /// <summary>
        /// Gets the kernel size.
        /// </summary>
        public int[] KernelSize => _kernelSize;

        /// <summary>
        /// Gets the stride.
        /// </summary>
        public int[] Stride => _stride;

        /// <summary>
        /// Gets the padding.
        /// </summary>
        public int[] Padding => _padding;

        /// <summary>
        /// Gets the dilation.
        /// </summary>
        public int[] Dilation => _dilation;

        /// <summary>
        /// Computes the output shape for 2D convolution.
        /// </summary>
        /// <param name="inputShapes">The shapes of the inputs [input, weight].</param>
        /// <returns>The shape of the output tensor.</returns>
        public Shapes.SymbolicShape GetOutputShape(System.Collections.Generic.List<Shapes.SymbolicShape> inputShapes)
        {
            if (inputShapes == null || inputShapes.Count < 2)
            {
                throw new ArgumentException("Conv2D requires at least 2 input shapes (input and weight)");
            }

            var inputShape = inputShapes[0];  // [N, C_in, H_in, W_in]
            var weightShape = inputShapes[1]; // [C_out, C_in, K_H, K_W]

            if (inputShape.Rank != 4)
            {
                throw new ArgumentException(
                    $"Input shape must have rank 4, got {inputShape.Rank}");
            }

            if (weightShape.Rank != 4)
            {
                throw new ArgumentException(
                    $"Weight shape must have rank 4, got {weightShape.Rank}");
            }

            // Extract dimensions
            var batchDim = inputShape.GetDimension(0);
            var outChannelsDim = weightShape.GetDimension(0);

            // Compute output spatial dimensions
            var heightDim = ComputeOutputSpatialDim(
                inputShape.GetDimension(2),
                _kernelSize[0],
                _padding[0],
                _stride[0],
                _dilation[0]);

            var widthDim = ComputeOutputSpatialDim(
                inputShape.GetDimension(3),
                _kernelSize[1],
                _padding[1],
                _stride[1],
                _dilation[1]);

            return new Shapes.SymbolicShape(
                batchDim,
                outChannelsDim,
                heightDim,
                widthDim);
        }

        /// <summary>
        /// Computes the output spatial dimension for convolution.
        /// </summary>
        /// <param name="inputDim">The input dimension.</param>
        /// <param name="kernelSize">The kernel size.</param>
        /// <param name="padding">The padding.</param>
        /// <param name="stride">The stride.</param>
        /// <param name="dilation">The dilation.</param>
        /// <returns>The output dimension.</returns>
        private static Shapes.SymbolicDimension ComputeOutputSpatialDim(
            Shapes.SymbolicDimension inputDim,
            int kernelSize,
            int padding,
            int stride,
            int dilation)
        {
            // Output formula: floor((input + 2*padding - dilation*(kernel-1) - 1) / stride + 1)

            if (inputDim.IsKnown())
            {
                int inputSize = inputDim.Value!.Value;
                int effectiveKernel = dilation * (kernelSize - 1) + 1;
                int outputSize = (inputSize + 2 * padding - effectiveKernel) / stride + 1;
                return new Shapes.SymbolicDimension($"{inputDim.Name}_out", outputSize);
            }
            else
            {
                // Unknown input size - create symbolic dimension
                string outputName = $"{inputDim.Name}_out";
                int minValue = (inputDim.MinValue + 2 * padding - dilation * (kernelSize - 1) - 1) / stride + 1;
                minValue = Math.Max(0, minValue);  // Ensure non-negative

                return new Shapes.SymbolicDimension(outputName, null, minValue);
            }
        }

        /// <summary>
        /// Computes the output shapes for all outputs.
        /// </summary>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <returns>A list of output shapes (single element for Conv2D).</returns>
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

            if (gradientShape.Rank != 4)
            {
                throw new ArgumentException(
                    $"Gradient shape must have rank 4, got {gradientShape.Rank}");
            }
        }

        /// <summary>
        /// Computes gradients for 2D convolution.
        /// </summary>
        /// <param name="gradOutput">The gradient from downstream operations.</param>
        /// <param name="inputs">The input tensors [input, weight] (bias is optional).</param>
        /// <param name="context">The operation context containing saved intermediate values.</param>
        /// <returns>An array of gradients [dInput, dWeight, dBias?].</returns>
        public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
        {
            if (gradOutput == null)
                throw new ArgumentNullException(nameof(gradOutput));

            if (inputs == null || inputs.Length < 2)
            {
                throw new ArgumentException("Conv2D requires at least 2 inputs (input and weight)");
            }

            var inputTensor = inputs[0];
            var weightTensor = inputs[1];

            // Gradient for input: conv_transpose(gradOutput, weight)
            var dInput = ComputeGradientForInput(gradOutput, weightTensor);

            // Gradient for weight: conv(input, gradOutput) (with appropriate transposition)
            var dWeight = ComputeGradientForWeight(inputTensor, gradOutput);

            // Gradient for bias (if present)
            Tensor[] gradients = new Tensor[inputs.Length];
            gradients[0] = dInput;
            gradients[1] = dWeight;

            if (inputs.Length >= 3 && inputs[2] != null)
            {
                // Bias gradient: sum over batch and spatial dimensions
                gradients[2] = ComputeGradientForBias(gradOutput);
            }

            return gradients;
        }

        /// <summary>
        /// Computes the gradient for the input tensor.
        /// </summary>
        /// <param name="gradOutput">The gradient from downstream.</param>
        /// <param name="weight">The weight tensor.</param>
        /// <returns>The gradient for the input.</returns>
        private static Tensor ComputeGradientForInput(Tensor gradOutput, Tensor weight)
        {
            // dInput = conv_transpose(gradOutput, weight)
            // In practice, this would use a tensor library to perform the operations
            // For now, we return gradOutput as a placeholder
            // TODO: Implement proper gradient computation with tensor operations

            return gradOutput;
        }

        /// <summary>
        /// Computes the gradient for the weight tensor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="gradOutput">The gradient from downstream.</param>
        /// <returns>The gradient for the weight.</returns>
        private static Tensor ComputeGradientForWeight(Tensor input, Tensor gradOutput)
        {
            // dWeight = conv(input, gradOutput) (with appropriate arrangement)
            // In practice, this would use a tensor library to perform the operations
            // For now, we return gradOutput as a placeholder
            // TODO: Implement proper gradient computation with tensor operations

            return gradOutput;
        }

        /// <summary>
        /// Computes the gradient for the bias tensor.
        /// </summary>
        /// <param name="gradOutput">The gradient from downstream.</param>
        /// <returns>The gradient for the bias.</returns>
        private static Tensor ComputeGradientForBias(Tensor gradOutput)
        {
            // dBias = sum(gradOutput, axis=[0, 2, 3])  # sum over batch and spatial dimensions
            // In practice, this would use a tensor library to perform the operations
            // For now, we return gradOutput as a placeholder
            // TODO: Implement proper gradient computation with tensor operations

            return gradOutput;
        }
    }
}
