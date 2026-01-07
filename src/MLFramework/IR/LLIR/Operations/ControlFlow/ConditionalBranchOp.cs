namespace MLFramework.IR.LLIR.Operations.ControlFlow
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.LLIR.Values;

    /// <summary>
    /// Conditional branch operation in the Low-Level IR (LLIR).
    /// Transfers control to the true or false target block based on the condition.
    /// </summary>
    public class ConditionalBranchOp : IROperation
    {
        /// <summary>Gets the condition value.</summary>
        public LLIRValue Condition { get; }

        /// <summary>Gets the target block when condition is true.</summary>
        public IRBlock TrueTarget { get; }

        /// <summary>Gets the target block when condition is false.</summary>
        public IRBlock FalseTarget { get; }

        /// <summary>
        /// Initializes a new instance of the ConditionalBranchOp class.
        /// </summary>
        /// <param name="condition">The condition value.</param>
        /// <param name="trueTarget">The target block when condition is true.</param>
        /// <param name="falseTarget">The target block when condition is false.</param>
        public ConditionalBranchOp(LLIRValue condition, IRBlock trueTarget, IRBlock falseTarget)
            : base("cond_branch", IROpcode.ConditionalBranch, new[] { condition }, System.Array.Empty<MLFramework.IR.Types.IIRType>(), null)
        {
            Condition = condition ?? throw new System.ArgumentNullException(nameof(condition));
            TrueTarget = trueTarget ?? throw new System.ArgumentNullException(nameof(trueTarget));
            FalseTarget = falseTarget ?? throw new System.ArgumentNullException(nameof(falseTarget));
        }

        public override void Validate()
        {
            // Validate condition type and target blocks exist
        }

        public override IROperation Clone()
        {
            return new ConditionalBranchOp(Condition, TrueTarget, FalseTarget);
        }
    }
}
