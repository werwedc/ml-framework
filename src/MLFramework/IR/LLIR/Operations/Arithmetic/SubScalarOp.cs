namespace MLFramework.IR.LLIR.Operations.Arithmetic
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.LLIR.Values;

    /// <summary>
    /// Scalar subtraction operation in the Low-Level IR (LLIR).
    /// </summary>
    public class SubScalarOp : IROperation
    {
        /// <summary>Gets the left-hand operand.</summary>
        public LLIRValue Lhs { get; }

        /// <summary>Gets the right-hand operand.</summary>
        public LLIRValue Rhs { get; }

        /// <summary>Gets the result of the subtraction.</summary>
        public LLIRValue Result => Results[0] as LLIRValue;

        /// <summary>
        /// Initializes a new instance of the SubScalarOp class.
        /// </summary>
        /// <param name="lhs">The left-hand operand.</param>
        /// <param name="rhs">The right-hand operand.</param>
        /// <param name="result">The result value.</param>
        public SubScalarOp(LLIRValue lhs, LLIRValue rhs, LLIRValue result)
            : base("sub_scalar", IROpcode.SubScalar, new[] { lhs, rhs }, new[] { result.Type }, null)
        {
            Lhs = lhs ?? throw new System.ArgumentNullException(nameof(lhs));
            Rhs = rhs ?? throw new System.ArgumentNullException(nameof(rhs));
            Results[0] = result ?? throw new System.ArgumentNullException(nameof(result));
        }

        public override void Validate()
        {
            // Type validation can be added here
        }

        public override IROperation Clone()
        {
            return new SubScalarOp(Lhs, Rhs, Result);
        }
    }
}
