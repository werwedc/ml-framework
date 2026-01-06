using RitterFramework.Core.Tensor;
using MLFramework.Amp;
using MLFramework.Core;

namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// Extension methods for Tensor with AMP support
    /// </summary>
    public static class AmpTensorExtensions
    {
        /// <summary>
        /// Backward pass with automatic AMP handling
        /// </summary>
        /// <param name="tensor">The tensor to compute gradients for</param>
        /// <param name="lossScaler">The loss scaler (optional)</param>
        public static void BackwardAmp(this Tensor tensor, ILossScaler? lossScaler = null)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            BackwardAmp(tensor, retainGraph: false, lossScaler);
        }

        /// <summary>
        /// Backward pass with gradient retention and AMP handling
        /// </summary>
        /// <param name="tensor">The tensor to compute gradients for</param>
        /// <param name="retainGraph">Whether to retain the computation graph</param>
        /// <param name="lossScaler">The loss scaler (optional)</param>
        public static void BackwardAmp(
            this Tensor tensor,
            bool retainGraph,
            ILossScaler? lossScaler = null)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            // For now, we call the regular backward
            // In a full implementation, we would:
            // 1. Apply loss scaling if lossScaler is provided
            // 2. Compute gradients
            // 3. Unscale gradients
            // 4. Handle dtype conversions

            tensor.Backward();
        }

        /// <summary>
        /// Gets the gradients with AMP-aware dtype handling
        /// </summary>
        /// <param name="tensor">The tensor to get gradients from</param>
        /// <param name="lossScaler">The loss scaler (optional)</param>
        /// <returns>Gradient tensor with correct dtype</returns>
        public static Tensor GradAmp(this Tensor tensor, ILossScaler? lossScaler = null)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            var gradient = tensor.Gradient;

            if (gradient == null)
                throw new InvalidOperationException("Tensor has no gradient. Call Backward() first.");

            // Ensure gradient is in Float32 for optimizer
            if (!gradient.IsDtype(DataType.Float32) && lossScaler != null)
            {
                // Unscale and convert to Float32
                var unscaled = lossScaler.UnscaleGradient(gradient);
                return unscaled.Cast(DataType.Float32);
            }

            return gradient;
        }

        /// <summary>
        /// Checks if a tensor requires AMP-aware backward pass
        /// </summary>
        /// <param name="tensor">The tensor to check</param>
        /// <returns>True if AMP-aware backward is needed, false otherwise</returns>
        public static bool NeedsAmpBackward(this Tensor tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            // Check if tensor is in mixed precision
            return tensor.IsLowPrecision();
        }
    }
}
