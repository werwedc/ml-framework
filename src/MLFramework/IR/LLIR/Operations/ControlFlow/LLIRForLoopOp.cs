namespace MLFramework.IR.LLIR.Operations.ControlFlow
{
    using MLFramework.IR;
    using MLFramework.IR.Operations;
    using MLFramework.IR.LLIR.Values;

    /// <summary>
    /// Loop unroll hint for the Low-Level IR (LLIR) for loop operation.
    /// </summary>
    public enum LoopUnrollHint
    {
        /// <summary>No unrolling hint.</summary>
        None,
        /// <summary>Hint to unroll the loop.</summary>
        Unroll,
        /// <summary>Hint to unroll and jam the loop.</summary>
        UnrollAndJam
    }

    /// <summary>
    /// For loop operation in the Low-Level IR (LLIR).
    /// Represents a counted loop with explicit start, end, and step values.
    /// </summary>
    public class LLIRForLoopOp : IROperation
    {
        /// <summary>Gets the start value of the loop.</summary>
        public LLIRValue Start { get; }

        /// <summary>Gets the end value of the loop.</summary>
        public LLIRValue End { get; }

        /// <summary>Gets the step value of the loop.</summary>
        public LLIRValue Step { get; }

        /// <summary>Gets the induction variable for this loop.</summary>
        public LLIRValue InductionVariable { get; }

        /// <summary>Gets the body block of this loop.</summary>
        public IRBlock Body { get; }

        /// <summary>Gets the loop unroll hint.</summary>
        public LoopUnrollHint UnrollHint { get; }

        /// <summary>
        /// Initializes a new instance of the LLIRForLoopOp class.
        /// </summary>
        /// <param name="start">The start value of the loop.</param>
        /// <param name="end">The end value of the loop.</param>
        /// <param name="step">The step value of the loop.</param>
        /// <param name="inductionVariable">The induction variable.</param>
        /// <param name="body">The body block of the loop.</param>
        /// <param name="unrollHint">The loop unroll hint.</param>
        public LLIRForLoopOp(LLIRValue start, LLIRValue end, LLIRValue step,
                            LLIRValue inductionVariable, IRBlock body,
                            LoopUnrollHint unrollHint = LoopUnrollHint.None)
            : base("for_loop", IROpcode.ForLoopOp, new[] { start, end, step, inductionVariable }, System.Array.Empty<MLFramework.IR.Types.IIRType>(), null)
        {
            Start = start ?? throw new System.ArgumentNullException(nameof(start));
            End = end ?? throw new System.ArgumentNullException(nameof(end));
            Step = step ?? throw new System.ArgumentNullException(nameof(step));
            InductionVariable = inductionVariable ?? throw new System.ArgumentNullException(nameof(inductionVariable));
            Body = body ?? throw new System.ArgumentNullException(nameof(body));
            UnrollHint = unrollHint;
        }

        public override void Validate()
        {
            // Validate start, end, step, and induction variable types
        }

        public override IROperation Clone()
        {
            return new LLIRForLoopOp(Start, End, Step, InductionVariable, Body, UnrollHint);
        }
    }
}
