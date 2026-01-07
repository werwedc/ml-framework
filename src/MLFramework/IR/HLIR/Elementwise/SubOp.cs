using System;

namespace MLFramework.IR.HLIR.Elementwise
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.Types;
    using MLFramework.IR.Values;

    /// <summary>
    /// Element-wise tensor subtraction operation.
    /// </summary>
    public class SubOp : IROperation
    {
        /// <summary>Gets the minuend (left operand).</summary>
        public IRValue Lhs { get; }

        /// <summary>Gets the subtrahend (right operand).</summary>
        public IRValue Rhs { get; }

        /// <summary>Gets the result of the subtraction.</summary>
        public IRValue Result => Results[0];

        /// <summary>
        /// Initializes a new instance of the SubOp class.
        /// </summary>
        /// <param name="lhs">The minuend.</param>
        /// <param name="rhs">The subtrahend.</param>
        /// <param name="result">The result value.</param>
        public SubOp(IRValue lhs, IRValue rhs, IRValue result)
            : base("sub", IROpcode.Sub, new[] { lhs, rhs }, new[] { result.Type }, null)
        {
            if (lhs == null)
                throw new ArgumentNullException(nameof(lhs));
            if (rhs == null)
                throw new ArgumentNullException(nameof(rhs));
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            Lhs = lhs;
            Rhs = rhs;
            Results[0] = result;
        }

        /// <summary>
        /// Validates the operation.
        /// </summary>
        public override void Validate()
        {
            if (Lhs.Type is not TensorType lhsType)
                throw new InvalidOperationException("LHS must be a tensor type");
            if (Rhs.Type is not TensorType rhsType)
                throw new InvalidOperationException("RHS must be a tensor type");
            if (Result.Type is not TensorType resultType)
                throw new InvalidOperationException("Result must be a tensor type");

            if (lhsType.ElementType != rhsType.ElementType)
                throw new InvalidOperationException($"Operand element types must match: {lhsType.ElementType} vs {rhsType.ElementType}");

            ValidateBroadcastShape(lhsType.Shape, rhsType.Shape);
        }

        private void ValidateBroadcastShape(int[] shape1, int[] shape2)
        {
            int maxRank = Math.Max(shape1.Length, shape2.Length);

            for (int i = 0; i < maxRank; i++)
            {
                int dim1 = GetDim(shape1, i, maxRank);
                int dim2 = GetDim(shape2, i, maxRank);

                if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
                {
                    throw new InvalidOperationException(
                        $"Incompatible shapes for broadcasting: {string.Join(", ", shape1)} and {string.Join(", ", shape2)}");
                }
            }
        }

        private int GetDim(int[] shape, int pos, int maxRank)
        {
            int idx = shape.Length - maxRank + pos;
            return idx < 0 ? 1 : shape[idx];
        }

        /// <summary>
        /// Creates a new SubOp with auto-generated result.
        /// </summary>
        public static IRValue Create(IRContext ctx, IRValue lhs, IRValue rhs, string name = null)
        {
            if (lhs == null)
                throw new ArgumentNullException(nameof(lhs));
            if (rhs == null)
                throw new ArgumentNullException(nameof(rhs));

            var lhsType = (TensorType)lhs.Type;
            var rhsType = (TensorType)rhs.Type;

            int[] broadcastShape = InferBroadcastShape(lhsType.Shape, rhsType.Shape);
            var resultType = new TensorType(lhsType.ElementType, broadcastShape);
            var result = ctx.CreateValue(resultType, name);

            var op = new SubOp(lhs, rhs, result);
            ctx.RegisterOperation(op);
            return result;
        }

        private static int[] InferBroadcastShape(int[] shape1, int[] shape2)
        {
            int maxRank = Math.Max(shape1.Length, shape2.Length);
            int[] result = new int[maxRank];

            for (int i = 0; i < maxRank; i++)
            {
                int dim1 = GetDimForInference(shape1, i, maxRank);
                int dim2 = GetDimForInference(shape2, i, maxRank);

                if (dim1 == dim2 || dim1 == 1 || dim2 == 1)
                {
                    result[i] = Math.Max(dim1, dim2);
                }
                else
                {
                    throw new InvalidOperationException($"Incompatible shapes: {string.Join(",", shape1)} and {string.Join(",", shape2)}");
                }
            }

            return result;
        }

        private static int GetDimForInference(int[] shape, int pos, int maxRank)
        {
            int idx = shape.Length - maxRank + pos;
            return idx < 0 ? 1 : shape[idx];
        }

        public override IROperation Clone()
        {
            return new SubOp(Lhs, Rhs, Result);
        }
    }
}
