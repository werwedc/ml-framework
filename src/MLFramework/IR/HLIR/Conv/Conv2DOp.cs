using System;

namespace MLFramework.IR.HLIR.Conv
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.Types;
    using MLFramework.IR.Values;

    /// <summary>
    /// 2D convolution operation.
    /// </summary>
    public class Conv2DOp : IROperation
    {
        /// <summary>Gets the input tensor (NCHW or NHWC format).</summary>
        public IRValue Input { get; }

        /// <summary>Gets the weight tensor.</summary>
        public IRValue Weight { get; }

        /// <summary>Gets the bias tensor (optional).</summary>
        public IRValue Bias { get; }

        /// <summary>Gets the output tensor.</summary>
        public IRValue Result => Results[0];

        /// <summary>Gets the kernel size.</summary>
        public int[] KernelSize { get; }

        /// <summary>Gets the stride.</summary>
        public int[] Stride { get; }

        /// <summary>Gets the padding.</summary>
        public int[] Padding { get; }

        /// <summary>Gets the dilation.</summary>
        public int[] Dilation { get; }

        /// <summary>Gets the number of groups for grouped convolution.</summary>
        public int Groups { get; }

        /// <summary>
        /// Initializes a new instance of the Conv2DOp class.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="weight">The weight tensor.</param>
        /// <param name="bias">The bias tensor (optional, can be null).</param>
        /// <param name="result">The result tensor.</param>
        /// <param name="kernelSize">The kernel size.</param>
        /// <param name="stride">The stride.</param>
        /// <param name="padding">The padding (optional, defaults to zero padding).</param>
        /// <param name="dilation">The dilation (optional, defaults to 1).</param>
        /// <param name="groups">The number of groups (optional, defaults to 1).</param>
        public Conv2DOp(IRValue input, IRValue weight, IRValue bias, IRValue result,
                         int[] kernelSize, int[] stride,
                         int[] padding = null, int[] dilation = null,
                         int groups = 1)
            : base("conv2d", IROpcode.Conv2D,
                    (bias != null) ? new[] { input, weight, bias } : new[] { input, weight },
                    new[] { result.Type }, null)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (weight == null)
                throw new ArgumentNullException(nameof(weight));
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            Input = input;
            Weight = weight;
            Bias = bias;
            KernelSize = kernelSize ?? throw new ArgumentNullException(nameof(kernelSize));
            Stride = stride ?? throw new ArgumentNullException(nameof(stride));
            Padding = padding ?? new int[] { 0, 0 };
            Dilation = dilation ?? new int[] { 1, 1 };
            Groups = groups;
            Results[0] = result;
        }

        /// <summary>
        /// Validates the operation.
        /// </summary>
        public override void Validate()
        {
            if (Input.Type is not TensorType inputType)
                throw new InvalidOperationException("Input must be a tensor type");
            if (Weight.Type is not TensorType weightType)
                throw new InvalidOperationException("Weight must be a tensor type");
            if (Result.Type is not TensorType resultType)
                throw new InvalidOperationException("Result must be a tensor type");

            if (inputType.Shape.Length != 4)
                throw new InvalidOperationException("Input must be 4D tensor (NCHW format)");

            if (weightType.Shape.Length != 4)
                throw new InvalidOperationException("Weight must be 4D tensor (OIHW format)");

            if (KernelSize.Length != 2)
                throw new InvalidOperationException("Kernel size must be 2-element array");

            if (Stride.Length != 2)
                throw new InvalidOperationException("Stride must be 2-element array");

            if (Padding.Length != 2)
                throw new InvalidOperationException("Padding must be 2-element array");

            if (Dilation.Length != 2)
                throw new InvalidOperationException("Dilation must be 2-element array");

            // Validate weight shape
            int inputChannels = inputType.Shape[1];
            int outputChannels = weightType.Shape[0];

            if (weightType.Shape[1] != inputChannels / Groups)
                throw new InvalidOperationException($"Weight input channels mismatch: expected {inputChannels / Groups}, got {weightType.Shape[1]}");

            if (weightType.Shape[2] != KernelSize[0])
                throw new InvalidOperationException($"Weight kernel height mismatch: expected {KernelSize[0]}, got {weightType.Shape[2]}");

            if (weightType.Shape[3] != KernelSize[1])
                throw new InvalidOperationException($"Weight kernel width mismatch: expected {KernelSize[1]}, got {weightType.Shape[3]}");

            // Validate bias shape
            if (Bias != null)
            {
                if (Bias.Type is not TensorType biasType)
                    throw new InvalidOperationException("Bias must be a tensor type");

                if (biasType.Shape.Length != 1)
                    throw new InvalidOperationException("Bias must be 1D tensor");

                if (biasType.Shape[0] != outputChannels)
                    throw new InvalidOperationException($"Bias size mismatch: expected {outputChannels}, got {biasType.Shape[0]}");
            }
        }

        /// <summary>
        /// Creates a new Conv2DOp with auto-generated result.
        /// </summary>
        public static IRValue Create(IRContext ctx, IRValue input, IRValue weight, IRValue bias,
                                  int[] kernelSize, int[] stride,
                                  int[] padding = null, int[] dilation = null,
                                  int groups = 1, string name = null)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (weight == null)
                throw new ArgumentNullException(nameof(weight));

            var inputType = (TensorType)input.Type;
            var weightType = (TensorType)weight.Type;

            // Infer output shape
            int[] outputShape = ComputeOutputShape(inputType.Shape, weightType.Shape, kernelSize, stride, padding, dilation);
            var resultType = new TensorType(inputType.ElementType, outputShape);
            var result = ctx.CreateValue(resultType, name);

            var op = new Conv2DOp(input, weight, bias, result, kernelSize, stride, padding, dilation, groups);
            ctx.RegisterOperation(op);
            return result;
        }

        /// <summary>
        /// Computes the output shape for the convolution.
        /// </summary>
        private static int[] ComputeOutputShape(int[] inputShape, int[] weightShape,
                                               int[] kernelSize, int[] stride,
                                               int[] padding, int[] dilation)
        {
            int batchSize = inputShape[0];
            int outputChannels = weightShape[0];
            int inputHeight = inputShape[2];
            int inputWidth = inputShape[3];

            // Compute output height and width
            int dilatedKernelHeight = (kernelSize[0] - 1) * dilation[0] + 1;
            int dilatedKernelWidth = (kernelSize[1] - 1) * dilation[1] + 1;

            int outputHeight = (inputHeight + 2 * padding[0] - dilatedKernelHeight) / stride[0] + 1;
            int outputWidth = (inputWidth + 2 * padding[1] - dilatedKernelWidth) / stride[1] + 1;

            return new int[] { batchSize, outputChannels, outputHeight, outputWidth };
        }

        /// <summary>
        /// Creates a clone of this operation.
        /// </summary>
        public override IROperation Clone()
        {
            return new Conv2DOp(Input, Weight, Bias, Result, KernelSize, Stride, Padding, Dilation, Groups);
        }
    }
}
