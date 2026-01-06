using System.Collections.Generic;
using MLFramework.Core;
using RitterFramework.Core.Tensor;

namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// Helper methods for AMP autograd integration
    /// Note: This is a stub implementation. Full implementation is in spec_amp_integration_autograd.md
    /// </summary>
    public static class AmpAutogradHelper
    {
        /// <summary>
        /// Converts gradients to the correct dtype for optimizer
        /// </summary>
        /// <param name="gradients">The gradients to convert</param>
        /// <param name="targetDtype">The target dtype (usually Float32)</param>
        /// <returns>Converted gradients</returns>
        public static Dictionary<string, Tensor> ConvertGradientsDtype(
            Dictionary<string, Tensor> gradients,
            DataType targetDtype)
        {
            var result = new Dictionary<string, Tensor>();

            foreach (var (name, grad) in gradients)
            {
                if (grad.IsDtype(targetDtype))
                {
                    result[name] = grad;
                }
                else
                {
                    result[name] = grad.Cast(targetDtype);
                }
            }

            return result;
        }

        /// <summary>
        /// Prepares gradients for optimizer step in AMP mode
        /// </summary>
        /// <param name="gradients">The gradients to prepare</param>
        /// <param name="lossScaler">The loss scaler</param>
        /// <param name="checkOverflow">Whether to check for overflow</param>
        /// <returns>True if gradients are valid, false if overflow detected</returns>
        public static bool PrepareGradientsForOptimizer(
            Dictionary<string, Tensor> gradients,
            ILossScaler lossScaler,
            bool checkOverflow = true)
        {
            if (checkOverflow && lossScaler != null)
            {
                bool hasOverflow = lossScaler.CheckOverflow(gradients);
                if (hasOverflow)
                {
                    return false;
                }
            }

            return true;
        }
    }
}
