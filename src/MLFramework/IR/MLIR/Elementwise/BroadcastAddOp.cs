namespace MLFramework.IR.MLIR.Elementwise
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.Types;
    using MLFramework.IR.Values;

    /// <summary>
    /// Broadcast-aware element-wise addition operation in the Mid-Level IR (MLIR).
    /// </summary>
    public class BroadcastAddOp : IROperation
    {
        /// <summary>Gets the left-hand operand.</summary>
        public IRValue Lhs { get; }

        /// <summary>Gets the right-hand operand.</summary>
        public IRValue Rhs { get; }

        /// <summary>Gets the result of the addition.</summary>
        public IRValue Result => Results[0];

        /// <summary>Gets the broadcast shape of the result.</summary>
        public int[] BroadcastShape { get; }

        /// <summary>
        /// Initializes a new instance of the BroadcastAddOp class.
        /// </summary>
        /// <param name="lhs">The left-hand operand.</param>
        /// <param name="rhs">The right-hand operand.</param>
        /// <param name="result">The result value.</param>
        /// <param name="broadcastShape">The broadcast shape of the result.</param>
        public BroadcastAddOp(IRValue lhs, IRValue rhs, IRValue result, int[] broadcastShape)
            : base("broadcast_add", IROpcode.Add, new[] { lhs, rhs }, new[] { result.Type }, null)
        {
            Lhs = lhs ?? throw new System.ArgumentNullException(nameof(lhs));
            Rhs = rhs ?? throw new System.ArgumentNullException(nameof(rhs));
            BroadcastShape = broadcastShape ?? throw new System.ArgumentNullException(nameof(broadcastShape));
            Results[0] = result ?? throw new System.ArgumentNullException(nameof(result));
        }

        public override void Validate()
        {
            // Validate broadcast shape and operand types
        }

        public override IROperation Clone()
        {
            return new BroadcastAddOp(Lhs, Rhs, Result, BroadcastShape);
        }
    }
}
