namespace MLFramework.IR.LLIR.Operations.ControlFlow
{
    using MLFramework.IR.Operations;

    /// <summary>
    /// Unconditional branch operation in the Low-Level IR (LLIR).
    /// Transfers control to the target block.
    /// </summary>
    public class BranchOp : IROperation
    {
        /// <summary>Gets the target block to branch to.</summary>
        public IRBlock Target { get; }

        /// <summary>
        /// Initializes a new instance of the BranchOp class.
        /// </summary>
        /// <param name="target">The target block to branch to.</param>
        public BranchOp(IRBlock target)
            : base("branch", IROpcode.Branch, System.Array.Empty<MLFramework.IR.Values.IRValue>(), System.Array.Empty<MLFramework.IR.Types.IIRType>(), null)
        {
            Target = target ?? throw new System.ArgumentNullException(nameof(target));
        }

        public override void Validate()
        {
            // Validate target block exists
        }

        public override IROperation Clone()
        {
            return new BranchOp(Target);
        }
    }
}
