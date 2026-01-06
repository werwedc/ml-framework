using System;
using System.Collections.Generic;
using System.Linq;
using RitterFramework.Core.Tensor;

namespace MLFramework.Amp
{
    /// <summary>
    /// Gradient clipping utilities for AMP
    /// </summary>
    public static class GradientClipper
    {
        /// <summary>
        /// Clips gradients by value (clamp between -clipValue and +clipValue)
        /// </summary>
        /// <param name="gradient">The gradient tensor to clip</param>
        /// <param name="clipValue">The maximum absolute value</param>
        /// <returns>Clipped gradient tensor</returns>
        public static Tensor ClipByValue(Tensor gradient, float clipValue)
        {
            if (gradient == null)
                throw new ArgumentNullException(nameof(gradient));

            if (clipValue < 0)
                throw new ArgumentException("Clip value must be non-negative", nameof(clipValue));

            if (clipValue == 0)
                throw new ArgumentException("Clip value cannot be zero", nameof(clipValue));

            // Create a new tensor with clipped values
            var data = gradient.Data;
            var clippedData = new float[data.Length];

            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] > clipValue)
                {
                    clippedData[i] = clipValue;
                }
                else if (data[i] < -clipValue)
                {
                    clippedData[i] = -clipValue;
                }
                else
                {
                    clippedData[i] = data[i];
                }
            }

            return new Tensor(clippedData, gradient.Shape, gradient.RequiresGrad);
        }

        /// <summary>
        /// Clips gradients by norm
        /// </summary>
        /// <param name="gradient">The gradient tensor to clip</param>
        /// <param name="maxNorm">The maximum L2 norm</param>
        /// <param name="normType">The type of norm (default: L2)</param>
        /// <returns>Clipped gradient tensor</returns>
        public static Tensor ClipByNorm(Tensor gradient, float maxNorm, float normType = 2.0f)
        {
            if (gradient == null)
                throw new ArgumentNullException(nameof(gradient));

            if (maxNorm <= 0)
                throw new ArgumentException("Max norm must be positive", nameof(maxNorm));

            if (normType <= 0)
                throw new ArgumentException("Norm type must be positive", nameof(normType));

            // Compute the norm
            float norm = ComputeNorm(gradient, normType);

            // If norm is already within bounds, return original gradient
            if (norm <= maxNorm)
                return gradient;

            // Calculate clipping scale factor
            float scale = maxNorm / norm;
            scale = MathF.Min(scale, 1.0f); // Don't increase gradients

            // Apply scaling
            return gradient * scale;
        }

        /// <summary>
        /// Clips multiple gradients by norm
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <param name="maxNorm">The maximum L2 norm</param>
        /// <param name="normType">The type of norm (default: L2)</param>
        /// <returns>Dictionary of clipped gradient tensors</returns>
        public static Dictionary<string, Tensor> ClipByNorm(
            Dictionary<string, Tensor> gradients,
            float maxNorm,
            float normType = 2.0f)
        {
            if (gradients == null)
                throw new ArgumentNullException(nameof(gradients));

            if (maxNorm <= 0)
                throw new ArgumentException("Max norm must be positive", nameof(maxNorm));

            if (normType <= 0)
                throw new ArgumentException("Norm type must be positive", nameof(normType));

            // Compute the total gradient norm
            float totalNorm = ComputeNorm(gradients, normType);

            // If norm is already within bounds, return original gradients
            if (totalNorm <= maxNorm)
                return gradients;

            // Calculate clipping scale factor
            float scale = maxNorm / totalNorm;
            scale = MathF.Min(scale, 1.0f); // Don't increase gradients

            // Apply scaling to all gradients
            var result = new Dictionary<string, Tensor>();
            foreach (var (name, grad) in gradients)
            {
                result[name] = grad * scale;
            }

            return result;
        }

        /// <summary>
        /// Computes the gradient norm for a single tensor
        /// </summary>
        /// <param name="gradient">The gradient tensor</param>
        /// <param name="normType">The type of norm (default: L2)</param>
        /// <returns>The gradient norm</returns>
        private static float ComputeNorm(Tensor gradient, float normType = 2.0f)
        {
            var data = gradient.Data;
            float norm = 0.0f;

            for (int i = 0; i < data.Length; i++)
            {
                norm += MathF.Pow(MathF.Abs(data[i]), normType);
            }

            return MathF.Pow(norm, 1.0f / normType);
        }

        /// <summary>
        /// Computes the gradient norm
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <param name="normType">The type of norm (default: L2)</param>
        /// <returns>The gradient norm</returns>
        public static float ComputeNorm(
            Dictionary<string, Tensor> gradients,
            float normType = 2.0f)
        {
            if (gradients == null)
                throw new ArgumentNullException(nameof(gradients));

            if (normType <= 0)
                throw new ArgumentException("Norm type must be positive", nameof(normType));

            float totalNorm = 0.0f;
            foreach (var grad in gradients.Values)
            {
                float norm = ComputeNorm(grad, normType);
                totalNorm += MathF.Pow(norm, normType);
            }
            return MathF.Pow(totalNorm, 1.0f / normType);
        }
    }
}
