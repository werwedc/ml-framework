using System;
using MLFramework.IR.Operations;
using MLFramework.IR.Types;
using MLFramework.IR.Values;

namespace MLFramework.IR.MLIR.Index
{
    /// <summary>
    /// Gather operation in MLIR.
    /// Gathers elements from the input tensor along a specified axis using indices.
    /// </summary>
    public class GatherOp : IROperation
    {
        /// <summary>Gets the input tensor.</summary>
        public IRValue Input => Operands[0];

        /// <summary>Gets the indices tensor.</summary>
        public IRValue Indices => Operands[1];

        /// <summary>Gets the result tensor.</summary>
        public IRValue Result => Results[0];

        /// <summary>Gets the axis along which to gather.</summary>
        public int Axis { get; }

        /// <summary>
        /// Initializes a new instance of the GatherOp class.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="result">The result value.</param>
        /// <param name="axis">The axis along which to gather.</param>
        public GatherOp(IRValue input, IRValue indices, IRValue result, int axis)
            : base("gather", IROpcode.Gather, new[] { input, indices }, new[] { result.Type }, null)
        {
            Results[0] = result ?? throw new ArgumentNullException(nameof(result));
            Axis = axis;
        }

        /// <summary>
        /// Validates the gather operation.
        /// </summary>
        public override void Validate()
        {
            // Validate that input is a tensor
            if (!(Input.Type is TensorType inputType))
                throw new InvalidOperationException($"Input must be a tensor type, got {Input.Type}");

            // Validate that indices is a tensor
            if (!(Indices.Type is TensorType indicesType))
                throw new InvalidOperationException($"Indices must be a tensor type, got {Indices.Type}");

            // Validate that indices type is integer
            if (indicesType.ElementType != DataType.Int32 && indicesType.ElementType != DataType.Int64)
                throw new InvalidOperationException($"Indices element type must be integer, got {indicesType.ElementType}");

            // Validate axis is within bounds
            int rank = inputType.Shape.Length;
            if (Axis < 0 || Axis >= rank)
                throw new InvalidOperationException($"Axis {Axis} out of bounds for tensor with rank {rank}");

            // Validate result type
            var expectedShape = ComputeOutputShape(inputType.Shape, indicesType.Shape);
            var expectedType = new TensorType(inputType.ElementType, expectedShape);
            if (Result.Type != expectedType)
                throw new InvalidOperationException($"Result type mismatch: expected {expectedType}, got {Result.Type}");
        }

        /// <summary>
        /// Computes the output shape of the gather operation.
        /// Output shape = input_shape[:axis] + indices_shape + input_shape[axis+1:]
        /// </summary>
        private int[] ComputeOutputShape(int[] inputShape, int[] indicesShape)
        {
            int[] outputShape = new int[inputShape.Length + indicesShape.Length - 1];
            int outputIndex = 0;

            // Copy dimensions before axis
            for (int i = 0; i < Axis; i++)
            {
                outputShape[outputIndex++] = inputShape[i];
            }

            // Copy indices dimensions
            for (int i = 0; i < indicesShape.Length; i++)
            {
                outputShape[outputIndex++] = indicesShape[i];
            }

            // Copy dimensions after axis
            for (int i = Axis + 1; i < inputShape.Length; i++)
            {
                outputShape[outputIndex++] = inputShape[i];
            }

            return outputShape;
        }

        /// <summary>
        /// Creates a clone of this operation.
        /// </summary>
        public override IROperation Clone()
        {
            return new GatherOp(Input, Indices, Result, Axis);
        }
    }
}
