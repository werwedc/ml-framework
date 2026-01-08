using System;
using MLFramework.IR.Operations;
using MLFramework.IR.Types;
using MLFramework.IR.Values;

namespace MLFramework.IR.MLIR.Type
{
    /// <summary>
    /// Type cast operation in MLIR.
    /// Converts a tensor from one data type to another.
    /// </summary>
    public class CastOp : IROperation
    {
        /// <summary>Gets the input tensor.</summary>
        public IRValue Input => Operands[0];

        /// <summary>Gets the result tensor (casted).</summary>
        public IRValue Result => Results[0];

        /// <summary>Gets the target data type.</summary>
        public DataType TargetType { get; }

        /// <summary>
        /// Initializes a new instance of the CastOp class.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="result">The result value.</param>
        /// <param name="targetType">The target data type.</param>
        public CastOp(IRValue input, IRValue result, DataType targetType)
            : base("cast", IROpcode.Cast, new[] { input }, new[] { result.Type }, null)
        {
            Results[0] = result ?? throw new ArgumentNullException(nameof(result));
            TargetType = targetType;
        }

        /// <summary>
        /// Validates the cast operation.
        /// </summary>
        public override void Validate()
        {
            // Validate that input is a tensor
            if (!(Input.Type is TensorType inputType))
                throw new InvalidOperationException($"Input must be a tensor type, got {Input.Type}");

            // Validate that result is a tensor
            if (!(Result.Type is TensorType resultType))
                throw new InvalidOperationException($"Result must be a tensor type, got {Result.Type}");

            // Validate that shapes are the same
            if (!ShapesEqual(inputType.Shape, resultType.Shape))
                throw new InvalidOperationException(
                    $"Input shape [{string.Join(",", inputType.Shape)}] must match result shape [{string.Join(",", resultType.Shape)}]");

            // Validate that result element type matches target type
            if (resultType.ElementType != TargetType)
                throw new InvalidOperationException(
                    $"Result element type {resultType.ElementType} must match target type {TargetType}");
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
            return new CastOp(Input, Result, TargetType);
        }
    }
}
