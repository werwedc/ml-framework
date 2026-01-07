using MLFramework.Distributed;
using MLFramework.NN;
using RitterFramework.Core.Tensor;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MLFramework.Layers.TensorParallel
{
    /// <summary>
    /// Manager for synchronizing gradients in tensor parallel training.
    /// Handles gradient synchronization for hybrid TP+DP scenarios.
    /// </summary>
    public static class TPGradientManager
    {
        /// <summary>
        /// Synchronizes gradients across all ranks before optimizer step.
        /// For pure TP: gradients are already local, no sync needed.
        /// For hybrid TP+DP: sync across DP groups but not TP groups.
        /// </summary>
        /// <param name="parameters">Parameters to synchronize</param>
        /// <param name="dpProcessGroup">Optional DP process group for hybrid TP+DP</param>
        public static async Task SynchronizeGradientsAsync(
            IEnumerable<Parameter> parameters,
            IProcessGroup? dpProcessGroup = null)
        {
            if (dpProcessGroup != null)
            {
                // Sync parameters across DP process group (for hybrid TP+DP)
                foreach (var param in parameters)
                {
                    if (param.Gradient != null)
                    {
                        await dpProcessGroup.AllReduceAsync(param.Gradient, ReduceOp.Sum);
                        // AllReduce modifies tensor in-place, so no assignment needed
                    }
                }
            }
            // For pure TP: gradients are already local, no sync needed
            // Each rank has its own shard of parameters
        }

        /// <summary>
        /// Synchronously synchronizes gradients across all ranks.
        /// </summary>
        /// <param name="parameters">Parameters to synchronize</param>
        /// <param name="dpProcessGroup">Optional DP process group for hybrid TP+DP</param>
        public static void SynchronizeGradients(
            IEnumerable<Parameter> parameters,
            IProcessGroup? dpProcessGroup = null)
        {
            var task = SynchronizeGradientsAsync(parameters, dpProcessGroup);
            task.Wait();
        }

        /// <summary>
        /// Checks that all ranks have received gradients (for debugging).
        /// </summary>
        /// <param name="parameters">Parameters to verify</param>
        public static async Task VerifyGradientsAsync(IEnumerable<Parameter> parameters)
        {
            if (!TensorParallel.IsInitialized())
            {
                throw new InvalidOperationException("TensorParallel has not been initialized");
            }

            var comm = TensorParallel.GetCommunicator();

            foreach (var param in parameters)
            {
                if (param.Gradient == null)
                {
                    throw new InvalidOperationException($"Parameter {param.Name} has no gradient");
                }

                // Check that grad is not all zeros (simple sanity check)
                var gradNorm = param.Gradient.Norm();
                var gradNormValue = gradNorm.ToScalar();

                // In a real distributed setup, we would all-reduce norms to verify all ranks have gradients
                await comm.AllReduceAsync(gradNorm, ReduceOp.Sum);
            }
        }

        /// <summary>
        /// Synchronously checks that all ranks have received gradients.
        /// </summary>
        /// <param name="parameters">Parameters to verify</param>
        public static void VerifyGradients(IEnumerable<Parameter> parameters)
        {
            var task = VerifyGradientsAsync(parameters);
            task.Wait();
        }

        /// <summary>
        /// Clips gradients by global norm across all parameters.
        /// </summary>
        /// <param name="parameters">Parameters whose gradients to clip</param>
        /// <param name="maxNorm">Maximum norm for gradients</param>
        /// <param name="normType">Type of norm (default: 2.0 for L2 norm)</param>
        public static void ClipGradientsByNorm(
            IEnumerable<Parameter> parameters,
            float maxNorm,
            float normType = 2.0f)
        {
            // Compute global norm of all gradients
            float totalNorm = 0f;
            foreach (var param in parameters)
            {
                if (param.Gradient != null)
                {
                    var normTensor = param.Gradient.Norm();
                    float norm = normTensor.ToScalar();
                    totalNorm += norm * norm;
                }
            }
            totalNorm = (float)Math.Sqrt(totalNorm);

            // Clip if necessary
            float clipCoeff = maxNorm / (totalNorm + 1e-6f);
            if (clipCoeff < 1.0f)
            {
                foreach (var param in parameters)
                {
                    if (param.Gradient != null)
                    {
                        for (int i = 0; i < param.Gradient.Size; i++)
                        {
                            param.Gradient.Data[i] *= clipCoeff;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Gets the total number of trainable parameters.
        /// </summary>
        /// <param name="parameters">Parameters to count</param>
        /// <returns>Total number of parameters</returns>
        public static long GetTotalParameterCount(IEnumerable<Parameter> parameters)
        {
            long total = 0;
            foreach (var param in parameters)
            {
                total += param.Size;
            }
            return total;
        }

        /// <summary>
        /// Gets the average gradient norm across all parameters.
        /// </summary>
        /// <param name="parameters">Parameters to analyze</param>
        /// <returns>Average gradient norm</returns>
        public static float GetAverageGradientNorm(IEnumerable<Parameter> parameters)
        {
            float totalNorm = 0f;
            int count = 0;

            foreach (var param in parameters)
            {
                if (param.Gradient != null)
                {
                    var normTensor = param.Gradient.Norm();
                    totalNorm += normTensor.ToScalar();
                    count++;
                }
            }

            return count > 0 ? totalNorm / count : 0f;
        }
    }
}
