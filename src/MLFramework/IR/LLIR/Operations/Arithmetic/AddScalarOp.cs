namespace MLFramework.IR.LLIR.Operations.Arithmetic
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.LLIR.Values;

    /// <summary>
    /// Scalar addition operation in the Low-Level IR (LLIR).
    /// </summary>
    public class AddScalarOp : IROperation
    {
        /// <summary>Gets the left-hand operand.</summary>
        public LLIRValue Lhs { get; }

        /// <summary>Gets the right-hand operand.</summary>
        public LLIRValue Rhs { get; }

        /// <summary>Gets the result of the addition.</summary>
        public LLIRValue Result => Results[0] as LLIRValue;

        /// <summary>
        /// Initializes a new instance of the AddScalarOp class.
        /// </summary>
        /// <param name="lhs">The left-hand operand.</param>
        /// <param name="rhs">The right-hand operand.</param>
        /// <param name="result">The result value.</param>
        public AddScalarOp(LLIRValue lhs, LLIRValue rhs, LLIRValue result)
            : base("add_scalar", IROpcode.Add, new[] { lhs, rhs }, new[] { result.Type }, null)
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
            return new AddScalarOp(Lhs, Rhs, Result);
        }
    }
}
