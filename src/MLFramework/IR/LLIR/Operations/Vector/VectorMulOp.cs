namespace MLFramework.IR.LLIR.Operations.Vector
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.LLIR.Values;

    /// <summary>
    /// Vector multiplication operation in the Low-Level IR (LLIR).
    /// Performs element-wise multiplication of two vectors.
    /// </summary>
    public class VectorMulOp : IROperation
    {
        /// <summary>Gets the left-hand vector operand.</summary>
        public LLIRValue Lhs { get; }

        /// <summary>Gets the right-hand vector operand.</summary>
        public LLIRValue Rhs { get; }

        /// <summary>Gets the result vector.</summary>
        public LLIRValue Result => Results[0] as LLIRValue;

        /// <summary>Gets the vector width (number of elements).</summary>
        public int VectorWidth { get; }

        /// <summary>
        /// Initializes a new instance of the VectorMulOp class.
        /// </summary>
        /// <param name="lhs">The left-hand vector operand.</param>
        /// <param name="rhs">The right-hand vector operand.</param>
        /// <param name="result">The result vector.</param>
        /// <param name="vectorWidth">The vector width (number of elements).</param>
        public VectorMulOp(LLIRValue lhs, LLIRValue rhs, LLIRValue result, int vectorWidth)
            : base("vector_mul", IROpcode.VectorMul, new[] { lhs, rhs }, new[] { result.Type }, null)
        {
            Lhs = lhs ?? throw new System.ArgumentNullException(nameof(lhs));
            Rhs = rhs ?? throw new System.ArgumentNullException(nameof(rhs));
            Results[0] = result ?? throw new System.ArgumentNullException(nameof(result));

            if (vectorWidth <= 0)
            {
                throw new System.ArgumentOutOfRangeException(nameof(vectorWidth), "Vector width must be positive.");
            }

            VectorWidth = vectorWidth;
        }

        public override void Validate()
        {
            // Type validation can be added here
        }

        public override IROperation Clone()
        {
            return new VectorMulOp(Lhs, Rhs, Result, VectorWidth);
        }
    }
}
