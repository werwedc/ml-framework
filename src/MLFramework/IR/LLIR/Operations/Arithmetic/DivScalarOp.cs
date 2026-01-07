namespace MLFramework.IR.LLIR.Operations.Arithmetic
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.LLIR.Values;

    /// <summary>
    /// Scalar division operation in the Low-Level IR (LLIR).
    /// </summary>
    public class DivScalarOp : IROperation
    {
        /// <summary>Gets the left-hand operand (dividend).</summary>
        public LLIRValue Lhs { get; }

        /// <summary>Gets the right-hand operand (divisor).</summary>
        public LLIRValue Rhs { get; }

        /// <summary>Gets the result of the division.</summary>
        public LLIRValue Result => Results[0] as LLIRValue;

        /// <summary>
        /// Initializes a new instance of the DivScalarOp class.
        /// </summary>
        /// <param name="lhs">The left-hand operand (dividend).</param>
        /// <param name="rhs">The right-hand operand (divisor).</param>
        /// <param name="result">The result value.</param>
        public DivScalarOp(LLIRValue lhs, LLIRValue rhs, LLIRValue result)
            : base("div_scalar", IROpcode.DivScalar, new[] { lhs, rhs }, new[] { result.Type }, null)
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
            return new DivScalarOp(Lhs, Rhs, Result);
        }
    }
}
