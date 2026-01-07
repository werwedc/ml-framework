using System;
using MLFramework.IR.Operations;
using MLFramework.IR.Types;
using MLFramework.IR.Values;

namespace MLFramework.IR.MLIR.Shape
{
    /// <summary>
    /// Dynamic reshape operation in MLIR.
    /// Reshapes a tensor to a new shape specified by a shape tensor.
    /// Unlike static reshape, the shape is provided as a runtime value.
    /// </summary>
    public class DynamicReshapeOp : IROperation
    {
        /// <summary>Gets the input tensor.</summary>
        public IRValue Input => Operands[0];

        /// <summary>Gets the shape tensor (1D tensor with the new shape).</summary>
        public IRValue Shape => Operands[1];

        /// <summary>Gets the result tensor (reshaped).</summary>
        public IRValue Result => Results[0];

        /// <summary>
        /// Initializes a new instance of the DynamicReshapeOp class.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="shape">The shape tensor (1D tensor of integers).</param>
        /// <param name="result">The result value.</param>
        public DynamicReshapeOp(IRValue input, IRValue shape, IRValue result)
            : base("dynamic_reshape", IROpcode.DynamicReshape, new[] { input, shape }, new[] { result.Type }, null)
        {
            Results[0] = result ?? throw new ArgumentNullException(nameof(result));
        }

        /// <summary>
        /// Validates the dynamic reshape operation.
        /// </summary>
        public override void Validate()
        {
            // Validate that input is a tensor
            if (!(Input.Type is TensorType inputType))
                throw new InvalidOperationException($"Input must be a tensor type, got {Input.Type}");

            // Validate that shape is a tensor
            if (!(Shape.Type is TensorType shapeType))
                throw new InvalidOperationException($"Shape must be a tensor type, got {Shape.Type}");

            // Validate that shape is 1D
            if (shapeType.Shape.Length != 1)
                throw new InvalidOperationException($"Shape must be 1D, got {shapeType.Shape.Length}D");

            // Validate that shape element type is integer
            if (shapeType.ElementType != DataType.Int32 && shapeType.ElementType != DataType.Int64)
                throw new InvalidOperationException($"Shape element type must be integer, got {shapeType.ElementType}");

            // Validate that result is a tensor
            if (!(Result.Type is TensorType resultType))
                throw new InvalidOperationException($"Result must be a tensor type, got {Result.Type}");

            // Validate that element types are the same
            if (inputType.ElementType != resultType.ElementType)
                throw new InvalidOperationException(
                    $"Input element type {inputType.ElementType} must match result element type {resultType.ElementType}");

            // Note: We cannot validate that the total number of elements matches
            // because the shape is a runtime value. This will be checked at runtime.
        }

        /// <summary>
        /// Creates a clone of this operation.
        /// </summary>
        public override IROperation Clone()
        {
            return new DynamicReshapeOp(Input, Shape, Result);
        }
    }
}
