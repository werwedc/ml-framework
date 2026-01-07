using System;

namespace MLFramework.IR.HLIR.Matrix
{
    using MLFramework.IR.Operations;
    using MLFramework.IR.Types;
    using MLFramework.IR.Values;

    /// <summary>
    /// Matrix multiplication operation.
    /// </summary>
    public class MatMulOp : IROperation
    {
        /// <summary>Gets the left matrix operand.</summary>
        public IRValue Lhs { get; }

        /// <summary>Gets the right matrix operand.</summary>
        public IRValue Rhs { get; }

        /// <summary>Gets the result of the matrix multiplication.</summary>
        public IRValue Result => Results[0];

        /// <summary>Gets whether to transpose the left matrix.</summary>
        public bool TransposeA { get; }

        /// <summary>Gets whether to transpose the right matrix.</summary>
        public bool TransposeB { get; }

        /// <summary>
        /// Initializes a new instance of the MatMulOp class.
        /// </summary>
        /// <param name="lhs">The left matrix operand.</param>
        /// <param name="rhs">The right matrix operand.</param>
        /// <param name="result">The result value.</param>
        /// <param name="transposeA">Whether to transpose the left matrix.</param>
        /// <param name="transposeB">Whether to transpose the right matrix.</param>
        public MatMulOp(IRValue lhs, IRValue rhs, IRValue result, bool transposeA = false, bool transposeB = false)
            : base("matmul", IROpcode.MatMul, new[] { lhs, rhs }, new[] { result.Type }, null)
        {
            if (lhs == null)
                throw new ArgumentNullException(nameof(lhs));
            if (rhs == null)
                throw new ArgumentNullException(nameof(rhs));
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            Lhs = lhs;
            Rhs = rhs;
            TransposeA = transposeA;
            TransposeB = transposeB;
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

            // Get effective inner dimensions based on transpose flags
            int lhsInnerDim = TransposeA ? lhsType.Shape[lhsType.Shape.Length - 2] : lhsType.Shape[lhsType.Shape.Length - 1];
            int rhsInnerDim = TransposeB ? rhsType.Shape[rhsType.Shape.Length - 1] : rhsType.Shape[rhsType.Shape.Length - 2];

            if (lhsInnerDim != rhsInnerDim)
            {
                throw new InvalidOperationException(
                    $"Inner dimensions must match for matrix multiplication: {lhsInnerDim} vs {rhsInnerDim}");
            }
        }

        /// <summary>
        /// Creates a new MatMulOp with auto-generated result.
        /// </summary>
        public static IRValue Create(IRContext ctx, IRValue lhs, IRValue rhs,
                                  bool transposeA = false, bool transposeB = false,
                                  string name = null)
        {
            if (lhs == null)
                throw new ArgumentNullException(nameof(lhs));
            if (rhs == null)
                throw new ArgumentNullException(nameof(rhs));

            var lhsType = (TensorType)lhs.Type;
            var rhsType = (TensorType)rhs.Type;

            // Infer output shape
            int[] outputShape = InferOutputShape(lhsType.Shape, rhsType.Shape, transposeA, transposeB);
            var resultType = new TensorType(lhsType.ElementType, outputShape);
            var result = ctx.CreateValue(resultType, name);

            var op = new MatMulOp(lhs, rhs, result, transposeA, transposeB);
            ctx.RegisterOperation(op);
            return result;
        }

        /// <summary>
        /// Infers the output shape for matrix multiplication.
        /// </summary>
        private static int[] InferOutputShape(int[] lhsShape, int[] rhsShape, bool transposeA, bool transposeB)
        {
            int lhsRank = lhsShape.Length;
            int rhsRank = rhsShape.Length;

            if (lhsRank < 2 || rhsRank < 2)
                throw new InvalidOperationException("Matrix multiplication requires at least 2D tensors");

            // Get outer dimensions (batch dimensions)
            int batchRank = Math.Max(lhsRank - 2, rhsRank - 2);
            int[] batchShape = new int[batchRank];

            for (int i = 0; i < batchRank; i++)
            {
                int lhsIdx = i - (batchRank - (lhsRank - 2));
                int rhsIdx = i - (batchRank - (rhsRank - 2));

                int lhsDim = (lhsIdx >= 0) ? lhsShape[lhsIdx] : 1;
                int rhsDim = (rhsIdx >= 0) ? rhsShape[rhsIdx] : 1;

                // Check broadcasting compatibility
                if (lhsDim != rhsDim && lhsDim != 1 && rhsDim != 1)
                {
                    throw new InvalidOperationException($"Incompatible batch dimensions: {lhsDim} vs {rhsDim}");
                }

                batchShape[i] = Math.Max(lhsDim, rhsDim);
            }

            // Get matrix dimensions
            int m = transposeA ? lhsShape[lhsRank - 1] : lhsShape[lhsRank - 2];
            int n = transposeB ? rhsShape[rhsRank - 2] : rhsShape[rhsRank - 1];

            int[] outputShape = new int[batchRank + 2];
            Array.Copy(batchShape, 0, outputShape, 0, batchRank);
            outputShape[batchRank] = m;
            outputShape[batchRank + 1] = n;

            return outputShape;
        }

        /// <summary>
        /// Creates a clone of this operation.
        /// </summary>
        public override IROperation Clone()
        {
            return new MatMulOp(Lhs, Rhs, Result, TransposeA, TransposeB);
        }
    }
}
