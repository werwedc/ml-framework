using System;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Quantization.Calibration
{
    /// <summary>
    /// Calibrator that maintains moving averages of min/max values.
    /// Useful for dynamic ranges and smooths out fluctuations in activation distributions.
    /// </summary>
    public class MovingAverageCalibrator : ICalibrator
    {
        private readonly int _windowSize;
        private readonly bool _symmetric;
        private readonly int _quantMin;
        private readonly int _quantMax;
        private readonly Queue<float> _minWindow;
        private readonly Queue<float> _maxWindow;
        private bool _hasCollectedData;

        /// <summary>
        /// Initializes a new instance of the MovingAverageCalibrator class.
        /// </summary>
        /// <param name="windowSize">Number of samples to maintain in the moving window.</param>
        /// <param name="symmetric">If true, uses symmetric quantization (zero-point = 0).</param>
        /// <param name="quantMin">Minimum quantized value (default: -128 for Int8).</param>
        /// <param name="quantMax">Maximum quantized value (default: 127 for Int8).</param>
        public MovingAverageCalibrator(int windowSize = 100, bool symmetric = false, int quantMin = -128, int quantMax = 127)
        {
            if (windowSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(windowSize), "Window size must be positive.");
            }

            _windowSize = windowSize;
            _symmetric = symmetric;
            _quantMin = quantMin;
            _quantMax = quantMax;
            _minWindow = new Queue<float>(windowSize);
            _maxWindow = new Queue<float>(windowSize);
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

            float batchMin = validData.Min();
            float batchMax = validData.Max();

            // Add to windows
            _minWindow.Enqueue(batchMin);
            _maxWindow.Enqueue(batchMax);

            // Remove oldest values if window is full
            if (_minWindow.Count > _windowSize)
            {
                _minWindow.Dequeue();
            }
            if (_maxWindow.Count > _windowSize)
            {
                _maxWindow.Dequeue();
            }

            _hasCollectedData = true;
        }

        /// <inheritdoc />
        public DataStructures.QuantizationParameters GetQuantizationParameters()
        {
            if (!_hasCollectedData)
            {
                throw new InvalidOperationException("No statistics have been collected. Call CollectStatistics first.");
            }

            // Calculate moving averages
            float minAvg = _minWindow.Average();
            float maxAvg = _maxWindow.Average();

            // Handle edge case: all values are the same
            if (Math.Abs(maxAvg - minAvg) < float.Epsilon)
            {
                // Set a small range around the value
                minAvg -= 1.0f;
                maxAvg += 1.0f;
            }

            float scale;
            int zeroPoint;

            if (_symmetric)
            {
                // Symmetric quantization
                scale = (float)Math.Max(Math.Abs(minAvg), Math.Abs(maxAvg)) / (_quantMax - _quantMin);
                zeroPoint = 0;
            }
            else
            {
                // Asymmetric quantization
                scale = (maxAvg - minAvg) / (_quantMax - _quantMin);
                zeroPoint = (int)Math.Round(_quantMin - minAvg / scale);

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
            _minWindow.Clear();
            _maxWindow.Clear();
            _hasCollectedData = false;
        }
    }
}
