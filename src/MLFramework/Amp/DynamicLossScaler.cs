using System;
using System.Collections.Generic;
using RitterFramework.Core.Tensor;

namespace MLFramework.Amp
{
    /// <summary>
    /// Dynamic loss scaler with automatic adjustment
    /// Prevents gradient underflow in FP16 training by adapting to overflow
    /// </summary>
    public class DynamicLossScaler : ILossScaler
    {
        private readonly float _initialScale;
        private readonly float _growthFactor;
        private readonly float _backoffFactor;
        private readonly int _growthInterval;
        private readonly float _minScale;
        private readonly float _maxScale;
        private readonly bool _enabled;

        private float _scale;
        private int _growthCounter;
        private int _totalOverflows;
        private int _totalSuccessfulIterations;
        private int _scaleIncreaseCount;
        private int _scaleDecreaseCount;
        private float _minScaleReached;
        private float _maxScaleReached;

        /// <summary>
        /// Gets the current scaling factor
        /// </summary>
        public float Scale => _scale;

        /// <summary>
        /// Gets whether the scaler is enabled
        /// </summary>
        public bool Enabled => _enabled;

        /// <summary>
        /// Gets the growth factor for increasing scale
        /// </summary>
        public float GrowthFactor => _growthFactor;

        /// <summary>
        /// Gets the backoff factor for decreasing scale
        /// </summary>
        public float BackoffFactor => _backoffFactor;

        /// <summary>
        /// Gets the number of consecutive iterations without overflow before increasing scale
        /// </summary>
        public int GrowthInterval => _growthInterval;

        /// <summary>
        /// Gets the minimum allowed scale
        /// </summary>
        public float MinScale => _minScale;

        /// <summary>
        /// Gets the maximum allowed scale
        /// </summary>
        public float MaxScale => _maxScale;

        /// <summary>
        /// Gets the number of consecutive iterations without overflow
        /// </summary>
        public int GrowthCounter => _growthCounter;

        /// <summary>
        /// Gets the total number of overflows encountered
        /// </summary>
        public int TotalOverflows => _totalOverflows;

        /// <summary>
        /// Creates a new DynamicLossScaler with default parameters
        /// </summary>
        public DynamicLossScaler()
            : this(65536.0f, 2.0f, 0.5f, 2000, 1.0f, 16777216.0f, true)
        {
        }

        /// <summary>
        /// Creates a new DynamicLossScaler with custom parameters
        /// </summary>
        /// <param name="initialScale">Initial scaling factor (default: 2^16 = 65536)</param>
        /// <param name="growthFactor">Factor to multiply scale when increasing (default: 2.0)</param>
        /// <param name="backoffFactor">Factor to multiply scale when decreasing (default: 0.5)</param>
        /// <param name="growthInterval">Iterations without overflow before increasing (default: 2000)</param>
        /// <param name="minScale">Minimum allowed scale (default: 1.0)</param>
        /// <param name="maxScale">Maximum allowed scale (default: 2^24 = 16777216)</param>
        /// <param name="enabled">Whether to enable scaling (default: true)</param>
        public DynamicLossScaler(
            float initialScale = 65536.0f,
            float growthFactor = 2.0f,
            float backoffFactor = 0.5f,
            int growthInterval = 2000,
            float minScale = 1.0f,
            float maxScale = 16777216.0f,
            bool enabled = true)
        {
            if (initialScale <= 0)
                throw new ArgumentException("Initial scale must be positive", nameof(initialScale));
            if (growthFactor <= 1.0f)
                throw new ArgumentException("Growth factor must be greater than 1.0", nameof(growthFactor));
            if (backoffFactor <= 0.0f || backoffFactor >= 1.0f)
                throw new ArgumentException("Backoff factor must be between 0 and 1", nameof(backoffFactor));
            if (growthInterval <= 0)
                throw new ArgumentException("Growth interval must be positive", nameof(growthInterval));
            if (minScale <= 0)
                throw new ArgumentException("Min scale must be positive", nameof(minScale));
            if (maxScale <= minScale)
                throw new ArgumentException("Max scale must be greater than min scale", nameof(maxScale));
            if (initialScale < minScale || initialScale > maxScale)
                throw new ArgumentException("Initial scale must be between min and max scale", nameof(initialScale));

            _initialScale = initialScale;
            _growthFactor = growthFactor;
            _backoffFactor = backoffFactor;
            _growthInterval = growthInterval;
            _minScale = minScale;
            _maxScale = maxScale;
            _enabled = enabled;

            Reset();
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

            if (!_enabled || _scale == 1.0f)
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

            if (!_enabled || _scale == 1.0f)
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

            if (!_enabled || _scale == 1.0f)
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

            if (!_enabled)
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

            if (!_enabled)
                return false;

            // Check for NaN or Inf values
            return HasOverflow(gradient);
        }

        /// <summary>
        /// Updates the scale factor based on overflow detection
        /// </summary>
        /// <param name="overflow">Whether overflow was detected</param>
        /// <returns>True if the step should be skipped, false otherwise</returns>
        public bool UpdateScale(bool overflow)
        {
            if (!_enabled)
                return false;

            if (overflow)
            {
                // Decrease scale by backoff factor
                float newScale = _scale * _backoffFactor;
                _scale = Math.Max(newScale, _minScale);
                _growthCounter = 0;
                _totalOverflows++;
                _scaleDecreaseCount++;

                // Update min/max scale reached
                if (_scale < _minScaleReached)
                    _minScaleReached = _scale;

                // Return true to indicate step should be skipped
                return true;
            }
            else
            {
                // Increment growth counter and successful iterations
                _growthCounter++;
                _totalSuccessfulIterations++;

                // Increase scale after growthInterval consecutive successes
                if (_growthCounter >= _growthInterval)
                {
                    float newScale = _scale * _growthFactor;
                    _scale = Math.Min(newScale, _maxScale);
                    _growthCounter = 0;
                    _scaleIncreaseCount++;

                    // Update min/max scale reached
                    if (_scale > _maxScaleReached)
                        _maxScaleReached = _scale;
                }

                // Return false to indicate step should proceed
                return false;
            }
        }

        /// <summary>
        /// Gets the scale as a tensor for loss multiplication
        /// </summary>
        /// <returns>Scale value as a scalar tensor</returns>
        public Tensor GetScaleTensor()
        {
            if (!_enabled)
                throw new InvalidOperationException("Scaler is disabled");

            return new Tensor(new float[] { _scale }, new int[] { 1 });
        }

        /// <summary>
        /// Gets the inverse scale for gradient unscaling
        /// </summary>
        /// <returns>Inverse scale as a scalar tensor</returns>
        public Tensor GetInverseScaleTensor()
        {
            if (!_enabled)
                throw new InvalidOperationException("Scaler is disabled");

            return new Tensor(new float[] { 1.0f / _scale }, new int[] { 1 });
        }

        /// <summary>
        /// Resets the scaler to initial state
        /// </summary>
        public void Reset()
        {
            _scale = _initialScale;
            _growthCounter = 0;
            _totalOverflows = 0;
            _totalSuccessfulIterations = 0;
            _scaleIncreaseCount = 0;
            _scaleDecreaseCount = 0;
            _minScaleReached = _initialScale;
            _maxScaleReached = _initialScale;
        }

        /// <summary>
        /// Gets statistics about the scaler's performance
        /// </summary>
        public DynamicScalerStats GetStats()
        {
            return new DynamicScalerStats(
                currentScale: _scale,
                totalOverflows: _totalOverflows,
                totalSuccessfulIterations: _totalSuccessfulIterations,
                scaleIncreaseCount: _scaleIncreaseCount,
                scaleDecreaseCount: _scaleDecreaseCount,
                minScaleReached: _minScaleReached,
                maxScaleReached: _maxScaleReached);
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
