namespace MLFramework.IR.MLIR.Reduce
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.Values;

    /// <summary>
    /// Reduction operation in the Mid-Level IR (MLIR).
    /// </summary>
    public class ReduceOp : IROperation
    {
        /// <summary>Gets the input tensor.</summary>
        public IRValue Input { get; }

        /// <summary>Gets the result tensor.</summary>
        public IRValue Result => Results[0];

        /// <summary>Gets the kind of reduction.</summary>
        public ReductionKind Kind { get; }

        /// <summary>Gets the axes to reduce along.</summary>
        public int[] Axes { get; }

        /// <summary>Gets whether to keep the reduced dimensions.</summary>
        public bool KeepDims { get; }

        /// <summary>
        /// Initializes a new instance of the ReduceOp class.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="result">The result tensor.</param>
        /// <param name="kind">The kind of reduction.</param>
        /// <param name="axes">The axes to reduce along.</param>
        /// <param name="keepDims">Whether to keep the reduced dimensions.</param>
        public ReduceOp(IRValue input, IRValue result, ReductionKind kind,
                       int[] axes, bool keepDims)
            : base("reduce", IROpcode.ReduceSum, new[] { input }, new[] { result.Type }, null)
        {
            Input = input ?? throw new System.ArgumentNullException(nameof(input));
            Kind = kind;
            Axes = axes ?? System.Array.Empty<int>();
            KeepDims = keepDims;
            Results[0] = result ?? throw new System.ArgumentNullException(nameof(result));
        }

        public override void Validate()
        {
            // Validate reduction parameters
        }

        public override IROperation Clone()
        {
            return new ReduceOp(Input, Result, Kind, Axes, KeepDims);
        }
    }
}
