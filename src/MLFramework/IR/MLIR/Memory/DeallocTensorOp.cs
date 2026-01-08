using System;
using MLFramework.IR.Operations;
using MLFramework.IR.Types;
using MLFramework.IR.Values;

namespace MLFramework.IR.MLIR.Memory
{
    /// <summary>
    /// Deallocates a previously allocated tensor in MLIR.
    /// This makes memory deallocation explicit in the IR.
    /// </summary>
    public class DeallocTensorOp : IROperation
    {
        /// <summary>Gets the tensor to deallocate.</summary>
        public IRValue Tensor => Operands[0];

        /// <summary>
        /// Initializes a new instance of the DeallocTensorOp class.
        /// </summary>
        /// <param name="tensor">The tensor to deallocate.</param>
        public DeallocTensorOp(IRValue tensor)
            : base("dealloc_tensor", IROpcode.DeallocTensor, new[] { tensor }, Array.Empty<IIRType>(), null)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
        }

        /// <summary>
        /// Validates the tensor deallocation operation.
        /// </summary>
        public override void Validate()
        {
            // Validate that the tensor is a tensor type
            if (!(Tensor.Type is TensorType))
                throw new InvalidOperationException($"Tensor must be a tensor type, got {Tensor.Type}");
        }

        /// <summary>
        /// Creates a clone of this operation.
        /// </summary>
        public override IROperation Clone()
        {
            return new DeallocTensorOp(Tensor);
        }
    }
}
