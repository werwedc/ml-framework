namespace MLFramework.IR.MLIR.Memory
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.Types;
    using MLFramework.IR.Values;

    /// <summary>
    /// Tensor allocation operation in the Mid-Level IR (MLIR).
    /// </summary>
    public class AllocTensorOp : IROperation
    {
        /// <summary>Gets the result tensor.</summary>
        public IRValue Result => Results[0];

        /// <summary>Gets the type of tensor being allocated.</summary>
        public TensorType AllocatedType { get; }

        /// <summary>
        /// Initializes a new instance of the AllocTensorOp class.
        /// </summary>
        /// <param name="result">The result value.</param>
        /// <param name="allocatedType">The type of tensor being allocated.</param>
        public AllocTensorOp(IRValue result, TensorType allocatedType)
            : base("alloc_tensor", IROpcode.AllocTensor, System.Array.Empty<IRValue>(), new[] { allocatedType }, null)
        {
            AllocatedType = allocatedType ?? throw new System.ArgumentNullException(nameof(allocatedType));
            Results[0] = result ?? throw new System.ArgumentNullException(nameof(result));
        }

        public override void Validate()
        {
            // Validate allocated type
        }

        public override IROperation Clone()
        {
            return new AllocTensorOp(Result, AllocatedType);
        }
    }
}
