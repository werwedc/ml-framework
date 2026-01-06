using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;

namespace MLFramework.Amp
{
    /// <summary>
    /// High-level GradScaler API for training with mixed precision.
    /// Wraps ILossScaler with convenience methods.
    /// </summary>
    public class GradScaler
    {
        private readonly ILossScaler _scaler;

        /// <summary>
        /// Gets the underlying loss scaler
        /// </summary>
        public ILossScaler Scaler => _scaler;

        /// <summary>
        /// Gets the current scaling factor
        /// </summary>
        public float Scale => _scaler.Scale;

        /// <summary>
        /// Gets whether the scaler is enabled
        /// </summary>
        public bool Enabled => _scaler.Enabled;

        /// <summary>
        /// Creates a GradScaler with a static loss scaler
        /// </summary>
        /// <param name="scale">Constant scaling factor (default: 2^16 = 65536)</param>
        /// <param name="enabled">Whether to enable scaling (default: true)</param>
        public GradScaler(float scale = 65536.0f, bool enabled = true)
        {
            _scaler = new StaticLossScaler(scale, enabled);
        }

        /// <summary>
        /// Creates a GradScaler with a dynamic loss scaler
        /// </summary>
        /// <param name="initialScale">Initial scaling factor</param>
        /// <param name="growthFactor">Factor to multiply scale when increasing</param>
        /// <param name="backoffFactor">Factor to multiply scale when decreasing</param>
        /// <param name="growthInterval">Iterations without overflow before increasing</param>
        /// <param name="minScale">Minimum allowed scale</param>
        /// <param name="maxScale">Maximum allowed scale</param>
        /// <param name="enabled">Whether to enable scaling</param>
        public GradScaler(
            float initialScale = 65536.0f,
            float growthFactor = 2.0f,
            float backoffFactor = 0.5f,
            int growthInterval = 2000,
            float minScale = 1.0f,
            float maxScale = 16777216.0f,
            bool enabled = true)
        {
            _scaler = new DynamicLossScaler(
                initialScale,
                growthFactor,
                backoffFactor,
                growthInterval,
                minScale,
                maxScale,
                enabled);
        }

        /// <summary>
        /// Creates a GradScaler with a custom ILossScaler implementation
        /// </summary>
        /// <param name="scaler">The loss scaler to wrap</param>
        public GradScaler(ILossScaler scaler)
        {
            _scaler = scaler ?? throw new ArgumentNullException(nameof(scaler));
        }

        /// <summary>
        /// Scales the loss tensor before backward pass
        /// </summary>
        /// <param name="loss">The loss tensor to scale</param>
        /// <returns>Scaled loss tensor</returns>
        public Tensor ScaleLoss(Tensor loss)
        {
            return _scaler.ScaleLoss(loss);
        }

        /// <summary>
        /// Unscales gradients manually (for custom optimizer logic)
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>Unscaled gradients</returns>
        public Dictionary<string, Tensor> Unscale(Dictionary<string, Tensor> gradients)
        {
            return _scaler.UnscaleGradients(gradients);
        }

        /// <summary>
        /// Updates the scale factor (for dynamic scalers)
        /// </summary>
        public void Update()
        {
            // For dynamic scalers, this is handled in the Step method
            // This is a no-op for compatibility
        }

        /// <summary>
        /// Checks for overflow in gradients
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>True if overflow detected, false otherwise</returns>
        public bool CheckOverflow(Dictionary<string, Tensor> gradients)
        {
            return _scaler.CheckOverflow(gradients);
        }

        /// <summary>
        /// Checks for overflow in a single gradient tensor
        /// </summary>
        /// <param name="gradient">The gradient tensor to check</param>
        /// <returns>True if overflow detected, false otherwise</returns>
        public bool CheckOverflow(Tensor gradient)
        {
            return _scaler.CheckOverflow(gradient);
        }

        /// <summary>
        /// Prepares gradients for optimizer step with overflow checking and unscaling
        /// </summary>
        /// <param name="gradients">The gradients to prepare</param>
        /// <returns>True if gradients are valid, false if overflow detected</returns>
        public bool PrepareGradients(Dictionary<string, Tensor> gradients)
        {
            bool hasOverflow = _scaler.CheckOverflow(gradients);
            _scaler.UpdateScale(hasOverflow);

            if (hasOverflow)
            {
                return false;
            }

            // Unscale gradients
            _scaler.UnscaleGradients(gradients);
            return true;
        }

        /// <summary>
        /// Prepares gradients and returns them (or null if overflow)
        /// </summary>
        /// <param name="gradients">The gradients to prepare</param>
        /// <returns>Unscaled gradients if valid, null if overflow detected</returns>
        public Dictionary<string, Tensor>? PrepareAndGetGradients(Dictionary<string, Tensor> gradients)
        {
            bool hasOverflow = _scaler.CheckOverflow(gradients);
            _scaler.UpdateScale(hasOverflow);

            if (hasOverflow)
            {
                return null;
            }

            return _scaler.UnscaleGradients(gradients);
        }

        /// <summary>
        /// Enables the scaler
        /// </summary>
        public void Enable()
        {
            // This depends on the implementation
            // DynamicLossScaler has an IsEnabled property that comes from options
            // We can't directly enable/disable it without recreating
            // For now, we'll document this limitation
            throw new NotImplementedException("Enable/Disable is not supported for this scaler implementation");
        }

        /// <summary>
        /// Disables the scaler
        /// </summary>
        public void Disable()
        {
            throw new NotImplementedException("Enable/Disable is not supported for this scaler implementation");
        }

        /// <summary>
        /// Resets the scaler (for dynamic scalers)
        /// </summary>
        public void Reset()
        {
            _scaler.Reset();
        }

        /// <summary>
        /// Gets the scale as a tensor for loss multiplication
        /// </summary>
        /// <returns>Scale value as a scalar tensor</returns>
        public Tensor GetScaleTensor()
        {
            return _scaler.GetScaleTensor();
        }

        /// <summary>
        /// Gets statistics (for dynamic scalers)
        /// </summary>
        /// <returns>Scaler statistics if available, null otherwise</returns>
        public DynamicScalerStats? GetStats()
        {
            return (_scaler as DynamicLossScaler)?.GetStats();
        }
    }
}
