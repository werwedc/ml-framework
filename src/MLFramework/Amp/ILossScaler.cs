using System.Collections.Generic;
using RitterFramework.Core.Tensor;

namespace MLFramework.Amp
{
    /// <summary>
    /// Interface for loss scalers used in Automatic Mixed Precision training
    /// </summary>
    public interface ILossScaler
    {
        /// <summary>
        /// Gets the current scaling factor
        /// </summary>
        float Scale { get; }

        /// <summary>
        /// Gets whether the scaler is enabled
        /// </summary>
        bool Enabled { get; }

        /// <summary>
        /// Scales the loss tensor before backward pass
        /// </summary>
        /// <param name="loss">The loss tensor to scale</param>
        /// <returns>Scaled loss tensor</returns>
        Tensor ScaleLoss(Tensor loss);

        /// <summary>
        /// Unscales gradients after backward pass
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>Unscaled gradients</returns>
        Dictionary<string, Tensor> UnscaleGradients(Dictionary<string, Tensor> gradients);

        /// <summary>
        /// Unscales a single gradient tensor
        /// </summary>
        /// <param name="gradient">The gradient tensor to unscale</param>
        /// <returns>Unscaled gradient</returns>
        Tensor UnscaleGradient(Tensor gradient);

        /// <summary>
        /// Checks for overflow in gradients
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>True if overflow detected, false otherwise</returns>
        bool CheckOverflow(Dictionary<string, Tensor> gradients);

        /// <summary>
        /// Checks for overflow in a single gradient tensor
        /// </summary>
        /// <param name="gradient">The gradient tensor to check</param>
        /// <returns>True if overflow detected, false otherwise</returns>
        bool CheckOverflow(Tensor gradient);

        /// <summary>
        /// Updates the scale factor based on overflow detection
        /// </summary>
        /// <param name="overflow">Whether overflow was detected</param>
        /// <returns>True if the step should be skipped, false otherwise</returns>
        bool UpdateScale(bool overflow);

        /// <summary>
        /// Resets the scaler to initial state
        /// </summary>
        void Reset();

        /// <summary>
        /// Gets the scale as a tensor for loss multiplication
        /// </summary>
        /// <returns>Scale value as a scalar tensor</returns>
        Tensor GetScaleTensor();

        /// <summary>
        /// Gets the inverse scale for gradient unscaling
        /// </summary>
        /// <returns>Inverse scale as a scalar tensor</returns>
        Tensor GetInverseScaleTensor();
    }
}
