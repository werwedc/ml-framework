using System;
using System.Linq;

namespace MLFramework.Quantization.Calibration
{
    /// <summary>
    /// Calibrator that uses min/max values to determine quantization parameters.
    /// Simple and fast, but sensitive to outliers.
    /// </summary>
    public class MinMaxCalibrator : ICalibrator
    {
        private float _min = float.MaxValue;
        private float _max = float.MinValue;
        private bool _hasCollectedData;
        private readonly bool _symmetric;
        private readonly int _quantMin;
        private readonly int _quantMax;

        /// <summary>
        /// Initializes a new instance of the MinMaxCalibrator class.
        /// </summary>
        /// <param name="symmetric">If true, uses symmetric quantization (zero-point = 0).</param>
        /// <param name="quantMin">Minimum quantized value (default: -128 for Int8).</param>
        /// <param name="quantMax">Maximum quantized value (default: 127 for Int8).</param>
        public MinMaxCalibrator(bool symmetric = false, int quantMin = -128, int quantMax = 127)
        {
            _symmetric = symmetric;
            _quantMin = quantMin;
            _quantMax = quantMax;
            _hasCollectedData = false;
        }

        /// <inheritdoc />
        public void CollectStatistics(float[] data)
        {
            if (data == null || data.Length == 0)
            {
                return;
            }

            // Filter out NaN and Inf values
            var validData = data.Where(v => !float.IsNaN(v) && !float.IsInfinity(v)).ToList();

            if (validData.Count == 0)
            {
                return;
            }

            _min = Math.Min(_min, validData.Min());
            _max = Math.Max(_max, validData.Max());
            _hasCollectedData = true;
        }

        /// <inheritdoc />
        public DataStructures.QuantizationParameters GetQuantizationParameters()
        {
            if (!_hasCollectedData)
            {
                throw new InvalidOperationException("No statistics have been collected. Call CollectStatistics first.");
            }

            // Handle edge case: all values are the same
            if (Math.Abs(_max - _min) < float.Epsilon)
            {
                // Set a small range around the value
                _min -= 1.0f;
                _max += 1.0f;
            }

            float scale;
            int zeroPoint;

            if (_symmetric)
            {
                // Symmetric quantization: zero-point is 0
                scale = (float)Math.Max(Math.Abs(_min), Math.Abs(_max)) / (_quantMax - _quantMin);
                zeroPoint = 0;
            }
            else
            {
                // Asymmetric quantization
                scale = (_max - _min) / (_quantMax - _quantMin);
                zeroPoint = (int)Math.Round(_quantMin - _min / scale);

                // Clamp zero-point to valid range
                zeroPoint = Math.Clamp(zeroPoint, _quantMin, _quantMax);
            }

            return new DataStructures.QuantizationParameters
            {
                Scale = scale,
                ZeroPoint = zeroPoint
            };
        }

        /// <inheritdoc />
        public void Reset()
        {
            _min = float.MaxValue;
            _max = float.MinValue;
            _hasCollectedData = false;
        }
    }
}
