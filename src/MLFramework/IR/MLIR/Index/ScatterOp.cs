using System;
using MLFramework.IR.Operations;
using MLFramework.IR.Types;
using MLFramework.IR.Values;

namespace MLFramework.IR.MLIR.Index
{
    /// <summary>
    /// Scatter operation in MLIR.
    /// Scatters updates into the input tensor along a specified axis using indices.
    /// </summary>
    public class ScatterOp : IROperation
    {
        /// <summary>Gets the input tensor.</summary>
        public IRValue Input => Operands[0];

        /// <summary>Gets the updates tensor.</summary>
        public IRValue Updates => Operands[1];

        /// <summary>Gets the indices tensor.</summary>
        public IRValue Indices => Operands[2];

        /// <summary>Gets the result tensor.</summary>
        public IRValue Result => Results[0];

        /// <summary>Gets the axis along which to scatter.</summary>
        public int Axis { get; }

        /// <summary>
        /// Initializes a new instance of the ScatterOp class.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="updates">The updates tensor.</param>
        /// <param name="indices">The indices tensor.</param>
        /// <param name="result">The result value.</param>
        /// <param name="axis">The axis along which to scatter.</param>
        public ScatterOp(IRValue input, IRValue updates, IRValue indices,
                        IRValue result, int axis)
            : base("scatter", IROpcode.Scatter, new[] { input, updates, indices }, new[] { result.Type }, null)
        {
            Results[0] = result ?? throw new ArgumentNullException(nameof(result));
            Axis = axis;
        }

        /// <summary>
        /// Validates the scatter operation.
        /// </summary>
        public override void Validate()
        {
            // Validate that input is a tensor
            if (!(Input.Type is TensorType inputType))
                throw new InvalidOperationException($"Input must be a tensor type, got {Input.Type}");

            // Validate that updates is a tensor
            if (!(Updates.Type is TensorType updatesType))
                throw new InvalidOperationException($"Updates must be a tensor type, got {Updates.Type}");

            // Validate that indices is a tensor
            if (!(Indices.Type is TensorType indicesType))
                throw new InvalidOperationException($"Indices must be a tensor type, got {Indices.Type}");

            // Validate that element types match between input and updates
            if (inputType.ElementType != updatesType.ElementType)
                throw new InvalidOperationException(
                    $"Input element type {inputType.ElementType} must match updates element type {updatesType.ElementType}");

            // Validate that indices type is integer
            if (indicesType.ElementType != DataType.Int32 && indicesType.ElementType != DataType.Int64)
                throw new InvalidOperationException($"Indices element type must be integer, got {indicesType.ElementType}");

            // Validate axis is within bounds
            int rank = inputType.Shape.Length;
            if (Axis < 0 || Axis >= rank)
                throw new InvalidOperationException($"Axis {Axis} out of bounds for tensor with rank {rank}");

            // Validate that updates shape matches expected shape
            // Expected: input_shape[:axis] + indices_shape + input_shape[axis+1:]
            int[] expectedUpdatesShape = ComputeUpdatesShape(inputType.Shape, indicesType.Shape);
            if (!ShapesEqual(updatesType.Shape, expectedUpdatesShape))
                throw new InvalidOperationException(
                    $"Updates shape mismatch: expected [{string.Join(",", expectedUpdatesShape)}], got [{string.Join(",", updatesType.Shape)}]");

            // Validate result type matches input type
            if (Result.Type != Input.Type)
                throw new InvalidOperationException($"Result type must match input type");
        }

        /// <summary>
        /// Computes the expected updates shape.
        /// </summary>
        private int[] ComputeUpdatesShape(int[] inputShape, int[] indicesShape)
        {
            int[] updatesShape = new int[inputShape.Length + indicesShape.Length - 1];
            int updatesIndex = 0;

            // Copy dimensions before axis
            for (int i = 0; i < Axis; i++)
            {
                updatesShape[updatesIndex++] = inputShape[i];
            }

            // Copy indices dimensions
            for (int i = 0; i < indicesShape.Length; i++)
            {
                updatesShape[updatesIndex++] = indicesShape[i];
            }

            // Copy dimensions after axis
            for (int i = Axis + 1; i < inputShape.Length; i++)
            {
                updatesShape[updatesIndex++] = inputShape[i];
            }

            return updatesShape;
        }

        /// <summary>
        /// Checks if two shapes are equal.
        /// </summary>
        private bool ShapesEqual(int[] shape1, int[] shape2)
        {
            if (shape1.Length != shape2.Length)
                return false;

            for (int i = 0; i < shape1.Length; i++)
            {
                if (shape1[i] != shape2[i])
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Creates a clone of this operation.
        /// </summary>
        public override IROperation Clone()
        {
            return new ScatterOp(Input, Updates, Indices, Result, Axis);
        }
    }
}
