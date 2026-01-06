using System.Collections.Generic;
using MLFramework.Core;
using MLFramework.Optimizers;
using RitterFramework.Core.Tensor;

namespace MLFramework.Amp.Integrations
{
    /// <summary>
    /// Extension methods for IOptimizer with AMP support
    /// </summary>
    public static class AmpOptimizerExtensions
    {
        /// <summary>
        /// Performs an optimizer step with automatic AMP handling
        /// </summary>
        /// <param name="optimizer">The optimizer</param>
        /// <param name="scaler">The GradScaler</param>
        /// <param name="gradients">The gradients (optional)</param>
        /// <returns>True if step was performed, false if skipped</returns>
        public static bool StepAmp(
            this IOptimizer optimizer,
            ILossScaler scaler,
            Dictionary<string, Tensor>? gradients = null)
        {
            if (gradients == null)
            {
                throw new ArgumentNullException(nameof(gradients));
            }

            return StepAmp(optimizer, scaler, gradients, checkOverflow: true, updateScale: true);
        }

        /// <summary>
        /// Performs an optimizer step with AMP and checks for overflow
        /// </summary>
        /// <param name="optimizer">The optimizer</param>
        /// <param name="scaler">The GradScaler</param>
        /// <param name="gradients">The gradients</param>
        /// <param name="checkOverflow">Whether to check for overflow</param>
        /// <param name="updateScale">Whether to update the loss scale</param>
        /// <returns>True if step was performed, false if skipped</returns>
        public static bool StepAmp(
            this IOptimizer optimizer,
            ILossScaler scaler,
            Dictionary<string, Tensor> gradients,
            bool checkOverflow,
            bool updateScale)
        {
            if (checkOverflow)
            {
                bool hasOverflow = scaler.CheckOverflow(gradients);
                if (updateScale)
                {
                    scaler.UpdateScale(hasOverflow);
                }

                if (hasOverflow)
                {
                    return false;
                }
            }

            var unscaledGrads = scaler.UnscaleGradients(gradients);
            optimizer.Step(unscaledGrads);

            return true;
        }

        /// <summary>
        /// Gets gradients with AMP-aware dtype handling
        /// </summary>
        /// <param name="optimizer">The optimizer</param>
        /// <param name="scaler">The GradScaler</param>
        /// <returns>Gradients with correct dtype</returns>
        public static Dictionary<string, Tensor> GetGradientsAmp(
            this IOptimizer optimizer,
            ILossScaler scaler)
        {
            // Note: IOptimizer doesn't have a GetGradients method
            // This would need to be added to the interface or handled differently
            throw new NotImplementedException("GetGradients is not implemented in IOptimizer");
        }

        /// <summary>
        /// Sets gradients with AMP-aware dtype handling
        /// </summary>
        /// <param name="optimizer">The optimizer</param>
        /// <param name="gradients">The gradients</param>
        /// <param name="targetDtype">The target dtype for gradients</param>
        public static void SetGradientsAmp(
            this IOptimizer optimizer,
            Dictionary<string, Tensor> gradients,
            DataType targetDtype)
        {
            // Note: IOptimizer doesn't have a SetGradients method
            // This would need to be added to the interface or handled differently
            throw new NotImplementedException("SetGradients is not implemented in IOptimizer");
        }
    }
}
