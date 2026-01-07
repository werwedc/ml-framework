using RitterFramework.Core.Tensor;
using MLFramework.Distributed;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLFramework.Distributed.FSDP
{
    /// <summary>
    /// Helper class for Reduce-Scatter operations in FSDP.
    /// </summary>
    public static class ReduceScatterHelper
    {
        /// <summary>
        /// Perform Reduce-Scatter on multiple gradients in parallel.
        /// </summary>
        /// <param name="processGroup">Process group for communication</param>
        /// <param name="gradients">Gradients to reduce and scatter</param>
        /// <param name="shardIndices">Shard index for each gradient</param>
        /// <param name="reduceOp">Reduction operation</param>
        /// <returns>List of reduced and scattered gradients</returns>
        public static Task<List<Tensor>> ReduceScatterMultipleAsync(
            IProcessGroup processGroup,
            List<Tensor> gradients,
            List<int> shardIndices,
            ReduceOp reduceOp = ReduceOp.Sum)
        {
            if (gradients == null || gradients.Count == 0)
                return Task.FromResult(new List<Tensor>());

            if (shardIndices == null)
                throw new ArgumentNullException(nameof(shardIndices));

            if (gradients.Count != shardIndices.Count)
                throw new ArgumentException("Gradients and shard indices must have the same count");

            var tasks = gradients.Zip(shardIndices, (grad, shardIdx) =>
            {
                var fullShape = grad.Shape.Select(dim => (long)dim).ToArray();
                var op = new ReduceScatterOperation(processGroup, fullShape, grad.Dtype, shardIdx, reduceOp);
                return op.ReduceScatterAsync(grad);
            }).ToList();

            return Task.WhenAll(tasks).ContinueWith(t => t.Result.ToList());
        }

        /// <summary>
        /// Verify that scattered gradients match the expected reduction.
        /// Used for testing.
        /// </summary>
        /// <param name="fullGradients">Full gradients from all devices</param>
        /// <param name="shardedGradients">Scattered gradients on each device</param>
        /// <param name="worldSize">Number of devices</param>
        /// <param name="reduceOp">Reduction operation</param>
        /// <returns>True if verification passes</returns>
        public static bool VerifyReduceScatter(
            List<Tensor> fullGradients,
            List<Tensor> shardedGradients,
            int worldSize,
            ReduceOp reduceOp = ReduceOp.Sum)
        {
            if (fullGradients == null || shardedGradients == null)
                return false;

            if (fullGradients.Count != shardedGradients.Count)
                return false;

            for (int i = 0; i < fullGradients.Count; i++)
            {
                var fullGrad = fullGradients[i];
                var shardedGrad = shardedGradients[i];

                var totalSize = fullGrad.Size;
                var chunkSize = (totalSize + worldSize - 1) / worldSize;

                // Each device should have its shard
                for (int rank = 0; rank < worldSize; rank++)
                {
                    var offset = rank * chunkSize;
                    var size = Math.Min(chunkSize, totalSize - offset);

                    for (int j = 0; j < size; j++)
                    {
                        var expected = fullGrad.Data[offset + j];
                        var actual = shardedGrad.Data[j];

                        if (reduceOp == ReduceOp.Avg)
                            expected /= worldSize;

                        if (Math.Abs(expected - actual) > 1e-5)
                            return false;
                    }
                }
            }

            return true;
        }
    }
}
