namespace MLFramework.IR.MLIR.Conv
{
    using MLFramework.IR;
    using MLFramework.IR.Operations;
    using MLFramework.IR.Values;

    /// <summary>
    /// Normalized convolution operation in the Mid-Level IR (MLIR).
    /// </summary>
    public class ConvOp : IROperation
    {
        /// <summary>Gets the input tensor.</summary>
        public IRValue Input { get; }

        /// <summary>Gets the weight tensor.</summary>
        public IRValue Weight { get; }

        /// <summary>Gets the result tensor.</summary>
        public IRValue Result => Results[0];

        /// <summary>Gets the input shape.</summary>
        public int[] InputShape { get; }

        /// <summary>Gets the weight shape.</summary>
        public int[] WeightShape { get; }

        /// <summary>Gets the output shape.</summary>
        public int[] OutputShape { get; }

        /// <summary>Gets the kernel size.</summary>
        public int[] KernelSize { get; }

        /// <summary>Gets the stride.</summary>
        public int[] Stride { get; }

        /// <summary>Gets the padding.</summary>
        public int[] Padding { get; }

        /// <summary>Gets the dilation.</summary>
        public int[] Dilation { get; }

        /// <summary>Gets the number of groups.</summary>
        public int Groups { get; }

        /// <summary>
        /// Initializes a new instance of the ConvOp class.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="weight">The weight tensor.</param>
        /// <param name="result">The result tensor.</param>
        /// <param name="inputShape">The input shape.</param>
        /// <param name="weightShape">The weight shape.</param>
        /// <param name="outputShape">The output shape.</param>
        /// <param name="kernelSize">The kernel size.</param>
        /// <param name="stride">The stride.</param>
        /// <param name="padding">The padding.</param>
        /// <param name="dilation">The dilation.</param>
        /// <param name="groups">The number of groups.</param>
        public ConvOp(IRValue input, IRValue weight, IRValue result,
                     int[] inputShape, int[] weightShape, int[] outputShape,
                     int[] kernelSize, int[] stride, int[] padding,
                     int[] dilation, int groups)
            : base("conv", IROpcode.Conv2D, new[] { input, weight }, new[] { result.Type }, null)
        {
            Input = input ?? throw new System.ArgumentNullException(nameof(input));
            Weight = weight ?? throw new System.ArgumentNullException(nameof(weight));
            InputShape = inputShape ?? throw new System.ArgumentNullException(nameof(inputShape));
            WeightShape = weightShape ?? throw new System.ArgumentNullException(nameof(weightShape));
            OutputShape = outputShape ?? throw new System.ArgumentNullException(nameof(outputShape));
            KernelSize = kernelSize ?? throw new System.ArgumentNullException(nameof(kernelSize));
            Stride = stride ?? throw new System.ArgumentNullException(nameof(stride));
            Padding = padding ?? throw new System.ArgumentNullException(nameof(padding));
            Dilation = dilation ?? throw new System.ArgumentNullException(nameof(dilation));
            Groups = groups;
            Results[0] = result ?? throw new System.ArgumentNullException(nameof(result));
        }

        public override void Validate()
        {
            // Validate convolution parameters
        }

        public override IROperation Clone()
        {
            return new ConvOp(Input, Weight, Result, InputShape, WeightShape, OutputShape,
                            KernelSize, Stride, Padding, Dilation, Groups);
        }
    }
}
