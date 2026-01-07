namespace MLFramework.IR.LLIR.Operations.Arithmetic
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.LLIR.Values;

    /// <summary>
    /// Scalar multiplication operation in the Low-Level IR (LLIR).
    /// </summary>
    public class MulScalarOp : IROperation
    {
        /// <summary>Gets the left-hand operand.</summary>
        public LLIRValue Lhs { get; }

        /// <summary>Gets the right-hand operand.</summary>
        public LLIRValue Rhs { get; }

        /// <summary>Gets the result of the multiplication.</summary>
        public LLIRValue Result => Results[0] as LLIRValue;

        /// <summary>
        /// Initializes a new instance of the MulScalarOp class.
        /// </summary>
        /// <param name="lhs">The left-hand operand.</param>
        /// <param name="rhs">The right-hand operand.</param>
        /// <param name="result">The result value.</param>
        public MulScalarOp(LLIRValue lhs, LLIRValue rhs, LLIRValue result)
            : base("mul_scalar", IROpcode.Mul, new[] { lhs, rhs }, new[] { result.Type }, null)
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
            return new MulScalarOp(Lhs, Rhs, Result);
        }
    }
}
