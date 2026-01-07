using MLFramework.Shapes;
using MLFramework.Shapes.Inference;

namespace MLFramework.Shapes.Inference.Rules
{
    /// <summary>
    /// Shape inference rule for matrix multiplication operations.
    /// Supports both 2D and batched matrix multiplication.
    /// </summary>
    public class MatMulRule : ShapeInferenceRuleBase
    {
        /// <summary>
        /// Gets the supported operations.
        /// </summary>
        protected override string[] SupportedOperations => new[] { "MatMul", "BatchMatMul", "Matmul", "Gemm" };

        /// <summary>
        /// Gets the expected input count.
        /// </summary>
        protected override int GetExpectedInputCount(string opName)
        {
            return 2;
        }

        /// <summary>
        /// Infers the output shape for matrix multiplication.
        /// </summary>
        /// <param name="opName">The name of the operation.</param>
        /// <param name="inputs">The input shapes.</param>
        /// <returns>The inferred output shape.</returns>
        protected override List<SymbolicShape> InferOutputShapes(string opName, IReadOnlyList<SymbolicShape> inputs)
        {
            var shapeA = inputs[0];
            var shapeB = inputs[1];

            // Handle 2D matrix multiplication: [M, K] @ [K, N] -> [M, N]
            if (shapeA.Rank == 2 && shapeB.Rank == 2)
            {
                var matDimM = shapeA.GetDimension(0);
                var matDimK1 = shapeA.GetDimension(1);
                var matDimK2 = shapeB.GetDimension(0);
                var matDimN = shapeB.GetDimension(1);

                // K dimensions must match (or be compatible through broadcasting)
                ValidateCompatibleDimensions(matDimK1, matDimK2, opName);

                return new List<SymbolicShape> { new SymbolicShape(matDimM, matDimN) };
            }

            // Handle batched matrix multiplication: [B1, ..., Bn, M, K] @ [B1, ..., Bn, K, N] -> [B1, ..., Bn, M, N]
            // Or: [M, K] @ [B1, ..., Bn, K, N] -> [B1, ..., Bn, M, N] (broadcasting on the left)
            // Or: [B1, ..., Bn, M, K] @ [K, N] -> [B1, ..., Bn, M, N] (broadcasting on the right)
            int maxRank = Math.Max(shapeA.Rank, shapeB.Rank);

            // Both shapes must be at least 2D for matmul
            if (shapeA.Rank < 2 || shapeB.Rank < 2)
            {
                throw new ArgumentException(
                    $"Both inputs for '{opName}' must have at least 2 dimensions. " +
                    $"Got ranks {shapeA.Rank} and {shapeB.Rank}");
            }

            // Get batch dimensions and matrix dimensions
            var (batchDimsA, matA) = ExtractBatchAndMatrixDims(shapeA);
            var (batchDimsB, matB) = ExtractBatchAndMatrixDims(shapeB);

            // Matrix dimensions: [M, K] @ [K, N]
            var batchDimM = matA[0];
            var batchDimK1 = matA[1];
            var batchDimK2 = matB[0];
            var batchDimN = matB[1];

            // K dimensions must match
            ValidateCompatibleDimensions(batchDimK1, batchDimK2, opName);

            // Broadcast batch dimensions
            var batchDims = BroadcastDimensions(batchDimsA, batchDimsB);

            // Combine batch dims with matrix dimensions [B1, ..., Bn, M, N]
            var outputDims = new List<SymbolicDimension>(batchDims);
            outputDims.Add(batchDimM);
            outputDims.Add(batchDimN);

            return new List<SymbolicShape> { new SymbolicShape(outputDims) };
        }

        /// <summary>
        /// Extracts batch dimensions and matrix dimensions from a shape.
        /// Returns (batchDims, [rowDim, colDim]) where rowDim and colDim are the last 2 dimensions.
        /// </summary>
        private (List<SymbolicDimension> batchDims, SymbolicDimension[] matrixDims) ExtractBatchAndMatrixDims(
            SymbolicShape shape)
        {
            if (shape.Rank == 2)
            {
                return (new List<SymbolicDimension>(), new[] { shape.GetDimension(0), shape.GetDimension(1) });
            }

            var batchDims = new List<SymbolicDimension>();
            for (int i = 0; i < shape.Rank - 2; i++)
            {
                batchDims.Add(shape.GetDimension(i));
            }

            var matrixDims = new[]
            {
                shape.GetDimension(shape.Rank - 2),
                shape.GetDimension(shape.Rank - 1)
            };

            return (batchDims, matrixDims);
        }

        /// <summary>
        /// Validates that two dimensions are compatible for matrix multiplication.
        /// </summary>
        private void ValidateCompatibleDimensions(SymbolicDimension dim1, SymbolicDimension dim2, string opName)
        {
            // Dimensions are compatible if:
            // 1. Both are concrete and equal
            // 2. One is 1 (broadcastable)
            // 3. At least one is symbolic (we assume they're compatible at runtime)

            bool dim1Known = dim1.IsKnown();
            bool dim2Known = dim2.IsKnown();

            if (dim1Known && dim2Known)
            {
                int val1 = dim1.Value!.Value;
                int val2 = dim2.Value!.Value;

                if (val1 != val2 && val1 != 1 && val2 != 1)
                {
                    throw new ArgumentException(
                        $"Incompatible dimensions for '{opName}': {val1} and {val2}. " +
                        $"For matrix multiplication, inner dimensions must match (or one must be 1 for broadcasting).");
                }
            }
            // If at least one is symbolic, we assume compatibility at runtime
        }

        /// <summary>
        /// Broadcasts two lists of dimensions.
        /// </summary>
        private List<SymbolicDimension> BroadcastDimensions(
            List<SymbolicDimension> dimsA, List<SymbolicDimension> dimsB)
        {
            // Align from right
            int maxLen = Math.Max(dimsA.Count, dimsB.Count);
            var result = new List<SymbolicDimension>();

            for (int i = 0; i < maxLen; i++)
            {
                int indexA = dimsA.Count - maxLen + i;
                int indexB = dimsB.Count - maxLen + i;

                SymbolicDimension dimA = (indexA >= 0) ? dimsA[indexA] : SymbolicDimensionFactory.Create("broadcast_dim", 1);
                SymbolicDimension dimB = (indexB >= 0) ? dimsB[indexB] : SymbolicDimensionFactory.Create("broadcast_dim", 1);

                result.Add(BroadcastDimension(dimA, dimB));
            }

            return result;
        }

        /// <summary>
        /// Broadcasts two dimensions according to NumPy broadcasting rules.
        /// </summary>
        private SymbolicDimension BroadcastDimension(SymbolicDimension dimA, SymbolicDimension dimB)
        {
            bool dimAKnown = dimA.IsKnown();
            bool dimBKnown = dimB.IsKnown();

            if (!dimAKnown && !dimBKnown)
            {
                // Both symbolic - keep the symbolic representation
                return dimA;
            }

            if (dimAKnown && !dimBKnown)
            {
                int valA = dimA.Value!.Value;
                return valA == 1 ? dimB : dimA;
            }

            if (!dimAKnown && dimBKnown)
            {
                int valB = dimB.Value!.Value;
                return valB == 1 ? dimA : dimB;
            }

            // Both are concrete
            int val1 = dimA.Value!.Value;
            int val2 = dimB.Value!.Value;

            if (val1 == val2)
                return dimA;

            if (val1 == 1)
                return dimB;

            if (val2 == 1)
                return dimA;

            throw new ArgumentException($"Cannot broadcast dimensions {val1} and {val2}");
        }
    }
}
