using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;

namespace MLFramework.Amp
{
    /// <summary>
    /// Static loss scaler with a constant scaling factor
    /// Prevents gradient underflow in FP16 training
    /// </summary>
    public class StaticLossScaler : ILossScaler
    {
        private readonly float _scale;
        private readonly Tensor? _scaleTensor;
        private readonly Tensor? _inverseScaleTensor;

        /// <summary>
        /// Gets the current scaling factor
        /// </summary>
        public float Scale => _scale;

        /// <summary>
        /// Gets whether the scaler is enabled
        /// </summary>
        public bool Enabled { get; }

        /// <summary>
        /// Creates a new StaticLossScaler
        /// </summary>
        /// <param name="scale">Constant scaling factor (default: 2^16 = 65536)</param>
        /// <param name="enabled">Whether to enable scaling (default: true)</param>
        public StaticLossScaler(float scale = 65536.0f, bool enabled = true)
        {
            if (scale <= 0)
                throw new ArgumentException("Scale must be positive", nameof(scale));

            _scale = scale;
            Enabled = enabled;

            if (enabled)
            {
                _scaleTensor = new Tensor(new float[] { scale }, new int[] { 1 });
                _inverseScaleTensor = new Tensor(new float[] { 1.0f / scale }, new int[] { 1 });
            }
        }

        /// <summary>
        /// Scales the loss tensor before backward pass
        /// </summary>
        /// <param name="loss">The loss tensor to scale</param>
        /// <returns>Scaled loss tensor</returns>
        public Tensor ScaleLoss(Tensor loss)
        {
            if (loss == null)
                throw new ArgumentNullException(nameof(loss));

            if (!Enabled || _scale == 1.0f)
                return loss;

            return loss * _scale;
        }

        /// <summary>
        /// Unscales gradients after backward pass
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>Unscaled gradients</returns>
        public Dictionary<string, Tensor> UnscaleGradients(Dictionary<string, Tensor> gradients)
        {
            if (gradients == null)
                throw new ArgumentNullException(nameof(gradients));

            if (!Enabled || _scale == 1.0f)
                return gradients;

            var unscaled = new Dictionary<string, Tensor>();
            foreach (var kvp in gradients)
            {
                unscaled[kvp.Key] = UnscaleGradient(kvp.Value);
            }

            return unscaled;
        }

        /// <summary>
        /// Unscales a single gradient tensor
        /// </summary>
        /// <param name="gradient">The gradient tensor to unscale</param>
        /// <returns>Unscaled gradient</returns>
        public Tensor UnscaleGradient(Tensor gradient)
        {
            if (gradient == null)
                throw new ArgumentNullException(nameof(gradient));

            if (!Enabled || _scale == 1.0f)
                return gradient;

            return gradient * (1.0f / _scale);
        }

        /// <summary>
        /// Checks for overflow in gradients
        /// </summary>
        /// <param name="gradients">Dictionary of parameter names to gradient tensors</param>
        /// <returns>True if overflow detected, false otherwise</returns>
        public bool CheckOverflow(Dictionary<string, Tensor> gradients)
        {
            if (gradients == null)
                throw new ArgumentNullException(nameof(gradients));

            if (!Enabled)
                return false;

            // Early exit on first detected overflow
            foreach (var grad in gradients.Values)
            {
                if (CheckOverflow(grad))
                    return true;
            }

            return false;
        }

        /// <summary>
        /// Checks for overflow in a single gradient tensor
        /// </summary>
        /// <param name="gradient">The gradient tensor to check</param>
        /// <returns>True if overflow detected, false otherwise</returns>
        public bool CheckOverflow(Tensor gradient)
        {
            if (gradient == null)
                throw new ArgumentNullException(nameof(gradient));

            if (!Enabled)
                return false;

            // Check for NaN or Inf values
            return HasOverflow(gradient);
        }

        /// <summary>
        /// Updates the scale based on overflow detection
        /// For static scaler, this is a no-op (scale remains constant)
        /// </summary>
        /// <param name="overflow">Whether overflow was detected</param>
        /// <returns>Always returns false (never skip steps in static mode)</returns>
        public bool UpdateScale(bool overflow)
        {
            // Static scaler doesn't adjust scale
            // Always return false to indicate step should proceed
            return false;
        }

        /// <summary>
        /// Resets the scaler to initial state
        /// For static scaler, this is a no-op
        /// </summary>
        public void Reset()
        {
            // Static scaler doesn't have state to reset
        }

        /// <summary>
        /// Gets the scale as a tensor for loss multiplication
        /// </summary>
        /// <returns>Scale value as a scalar tensor</returns>
        public Tensor GetScaleTensor()
        {
            if (!Enabled)
                throw new InvalidOperationException("Scaler is disabled");

            if (_scaleTensor == null)
                throw new InvalidOperationException("Scale tensor not initialized");

            return _scaleTensor.Clone();
        }

        /// <summary>
        /// Gets the inverse scale for gradient unscaling
        /// </summary>
        /// <returns>Inverse scale as a scalar tensor</returns>
        public Tensor GetInverseScaleTensor()
        {
            if (!Enabled)
                throw new InvalidOperationException("Scaler is disabled");

            if (_inverseScaleTensor == null)
                throw new InvalidOperationException("Inverse scale tensor not initialized");

            return _inverseScaleTensor.Clone();
        }

        #region Private Methods

        /// <summary>
        /// Checks if a tensor contains overflow (NaN or Inf)
        /// </summary>
        private bool HasOverflow(Tensor tensor)
        {
            int[] shape = tensor.Shape;
            int totalElements = 1;
            foreach (int dim in shape)
            {
                totalElements *= dim;
            }

            // Check each element for NaN or Infinity
            int[] indices = new int[shape.Length];
            for (int i = 0; i < totalElements; i++)
            {
                // Convert flat index to multi-dimensional indices
                int temp = i;
                for (int j = shape.Length - 1; j >= 0; j--)
                {
                    indices[j] = temp % shape[j];
                    temp /= shape[j];
                }

                float value = tensor[indices];

                // Check for NaN or Infinity
                if (float.IsNaN(value) || float.IsInfinity(value))
                {
                    return true;
                }
            }

            return false;
        }

        #endregion
    }
}
