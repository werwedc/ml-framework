using RitterFramework.Core.Tensor;

namespace MLFramework.Autograd.Operations
{
    /// <summary>
    /// Gradient function for matrix multiplication with dynamic shapes.
    /// Handles batch dimension broadcasting and symbolic batch sizes.
    /// </summary>
    public sealed class DynamicMatMulBackward : IDynamicGradientFunction
    {
        /// <summary>
        /// Gets the name of this operation.
        /// </summary>
        public string OperationName => "MatMul";

        /// <summary>
        /// Computes the output shape for matrix multiplication.
        /// </summary>
        /// <param name="inputShapes">The shapes of the input matrices [A, B].</param>
        /// <returns>The shape of the output matrix.</returns>
        public Shapes.SymbolicShape GetOutputShape(System.Collections.Generic.List<Shapes.SymbolicShape> inputShapes)
        {
            if (inputShapes == null || inputShapes.Count != 2)
            {
                throw new ArgumentException("MatMul requires exactly 2 input shapes");
            }

            var shapeA = inputShapes[0];
            var shapeB = inputShapes[1];

            // Handle batch dimensions
            // A: [..., M, K]
            // B: [..., K, N]
            // Output: [..., M, N]

            int rankA = shapeA.Rank;
            int rankB = shapeB.Rank;

            if (rankA < 2 || rankB < 2)
            {
                throw new ArgumentException(
                    "MatMul requires inputs with rank >= 2");
            }

            // Get batch dimensions
            int batchRankA = rankA - 2;
            int batchRankB = rankB - 2;
            int maxBatchRank = Math.Max(batchRankA, batchRankB);

            var outputDims = new List<Shapes.SymbolicDimension>();

            // Broadcast batch dimensions
            for (int i = 0; i < maxBatchRank; i++)
            {
                var dimA = batchRankA > i ? shapeA.GetDimension(i) : new Shapes.SymbolicDimension($"_batch_{i}", 1);
                var dimB = batchRankB > i ? shapeB.GetDimension(i) : new Shapes.SymbolicDimension($"_batch_{i}", 1);

                // Use broadcasting-compatible dimension
                if (dimA.IsKnown() && dimA.Value == 1)
                {
                    outputDims.Add(dimB);
                }
                else if (dimB.IsKnown() && dimB.Value == 1)
                {
                    outputDims.Add(dimA);
                }
                else if (dimA.Equals(dimB))
                {
                    outputDims.Add(dimA);
                }
                else
                {
                    // At least one is unknown, assume compatible
                    outputDims.Add(dimA.IsKnown() ? dimB : dimA);
                }
            }

            // Add matrix dimensions: M from A, N from B
            outputDims.Add(shapeA.GetDimension(rankA - 2));  // M
            outputDims.Add(shapeB.GetDimension(rankB - 1));  // N

            return new Shapes.SymbolicShape(outputDims);
        }

        /// <summary>
        /// Computes the output shapes for all outputs.
        /// </summary>
        /// <param name="inputShapes">The shapes of the input tensors.</param>
        /// <returns>A list of output shapes (single element for MatMul).</returns>
        public System.Collections.Generic.List<Shapes.SymbolicShape> GetOutputShapes(
            System.Collections.Generic.List<Shapes.SymbolicShape> inputShapes)
        {
            return new System.Collections.Generic.List<Shapes.SymbolicShape>
            {
                GetOutputShape(inputShapes)
            };
        }

        /// <summary>
        /// Validates that a gradient shape matches the expected input shape.
        /// </summary>
        /// <param name="gradientShape">The shape of the gradient tensor.</param>
        /// <exception cref="ArgumentException">Thrown when the gradient shape is invalid.</exception>
        public void ValidateGradientShape(Shapes.SymbolicShape gradientShape)
        {
            // Basic validation - shape should have at least 2 dimensions
            if (gradientShape == null)
            {
                throw new ArgumentNullException(nameof(gradientShape));
            }

            if (gradientShape.Rank < 2)
            {
                throw new ArgumentException(
                    $"Gradient shape must have rank >= 2, got {gradientShape.Rank}");
            }
        }

        /// <summary>
        /// Computes gradients for the matrix multiplication.
        /// </summary>
        /// <param name="gradOutput">The gradient from downstream operations.</param>
        /// <param name="inputs">The input tensors to the original operation.</param>
        /// <param name="context">The operation context containing saved intermediate values.</param>
        /// <returns>An array of gradients [dA, dB] for each input.</returns>
        public Tensor[] ComputeGrad(Tensor gradOutput, Tensor[] inputs, OperationContext context)
        {
            if (gradOutput == null)
                throw new ArgumentNullException(nameof(gradOutput));

            if (inputs == null || inputs.Length != 2)
            {
                throw new ArgumentException("MatMul requires exactly 2 inputs");
            }

            var A = inputs[0];
            var B = inputs[1];

            // Gradient for A: gradOutput * B^T
            // dA = gradOutput @ B^T
            var dA = ComputeGradientForA(gradOutput, B);

            // Gradient for B: A^T * gradOutput
            // dB = A^T @ gradOutput
            var dB = ComputeGradientForB(A, gradOutput);

            return new Tensor[] { dA, dB };
        }

        /// <summary>
        /// Computes the gradient for the first input matrix A.
        /// </summary>
        /// <param name="gradOutput">The gradient from downstream.</param>
        /// <param name="B">The second input matrix.</param>
        /// <returns>The gradient for A.</returns>
        private static Tensor ComputeGradientForA(Tensor gradOutput, Tensor B)
        {
            // dA = gradOutput @ B^T
            // In practice, this would use a tensor library to perform the operations
            // For now, we return gradOutput as a placeholder
            // TODO: Implement proper gradient computation with tensor operations

            return gradOutput;
        }

        /// <summary>
        /// Computes the gradient for the second input matrix B.
        /// </summary>
        /// <param name="A">The first input matrix.</param>
        /// <param name="gradOutput">The gradient from downstream.</param>
        /// <returns>The gradient for B.</returns>
        private static Tensor ComputeGradientForB(Tensor A, Tensor gradOutput)
        {
            // dB = A^T @ gradOutput
            // In practice, this would use a tensor library to perform the operations
            // For now, we return gradOutput as a placeholder
            // TODO: Implement proper gradient computation with tensor operations

            return gradOutput;
        }
    }
}
