namespace MLFramework.IR.MLIR.Loop
{
    using MLFramework.IR;
    using MLFramework.IR.Operations;
    using MLFramework.IR.Values;

    /// <summary>
    /// For loop operation in the Mid-Level IR (MLIR).
    /// </summary>
    public class ForLoopOp : IROperation
    {
        /// <summary>Gets the lower bound of the loop.</summary>
        public IRValue LowerBound { get; }

        /// <summary>Gets the upper bound of the loop.</summary>
        public IRValue UpperBound { get; }

        /// <summary>Gets the step value of the loop.</summary>
        public IRValue Step { get; }

        /// <summary>Gets the induction variable.</summary>
        public IRValue InductionVariable { get; }

        /// <summary>Gets the body block of the loop.</summary>
        public IRBlock Body { get; }

        /// <summary>
        /// Initializes a new instance of the ForLoopOp class.
        /// </summary>
        /// <param name="lowerBound">The lower bound.</param>
        /// <param name="upperBound">The upper bound.</param>
        /// <param name="step">The step value.</param>
        /// <param name="inductionVariable">The induction variable.</param>
        /// <param name="body">The body block.</param>
        public ForLoopOp(IRValue lowerBound, IRValue upperBound, IRValue step,
                        IRValue inductionVariable, IRBlock body)
            : base("for_loop", IROpcode.ForLoopOp, new[] { lowerBound, upperBound, step, inductionVariable }, System.Array.Empty<MLFramework.IR.Types.IIRType>(), null)
        {
            LowerBound = lowerBound ?? throw new System.ArgumentNullException(nameof(lowerBound));
            UpperBound = upperBound ?? throw new System.ArgumentNullException(nameof(upperBound));
            Step = step ?? throw new System.ArgumentNullException(nameof(step));
            InductionVariable = inductionVariable ?? throw new System.ArgumentNullException(nameof(inductionVariable));
            Body = body ?? throw new System.ArgumentNullException(nameof(body));
        }

        public override void Validate()
        {
            // Validate loop parameters
        }

        public override IROperation Clone()
        {
            return new ForLoopOp(LowerBound, UpperBound, Step, InductionVariable, Body);
        }
    }
}
