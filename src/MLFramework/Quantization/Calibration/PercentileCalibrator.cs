using System;
using System.Linq;

namespace MLFramework.Quantization.Calibration
{
    /// <summary>
    /// Calibrator that uses percentiles to exclude outliers from min/max calculations.
    /// More robust to extreme values than MinMaxCalibrator.
    /// </summary>
    public class PercentileCalibrator : ICalibrator
    {
        private readonly float _percentile;
        private readonly bool _symmetric;
        private readonly int _quantMin;
        private readonly int _quantMax;
        private System.Collections.Generic.List<float> _dataPoints;
        private bool _hasCollectedData;

        /// <summary>
        /// Initializes a new instance of the PercentileCalibrator class.
        /// </summary>
        /// <param name="percentile">Percentile value to use (default: 99.9 for excluding 0.1% outliers).</param>
        /// <param name="symmetric">If true, uses symmetric quantization (zero-point = 0).</param>
        /// <param name="quantMin">Minimum quantized value (default: -128 for Int8).</param>
        /// <param name="quantMax">Maximum quantized value (default: 127 for Int8).</param>
        public PercentileCalibrator(float percentile = 99.9f, bool symmetric = false, int quantMin = -128, int quantMax = 127)
        {
            if (percentile <= 50 || percentile >= 100)
            {
                throw new ArgumentOutOfRangeException(nameof(percentile), "Percentile must be between 50 and 100.");
            }

            _percentile = percentile;
            _symmetric = symmetric;
            _quantMin = quantMin;
            _quantMax = quantMax;
            _dataPoints = new System.Collections.Generic.List<float>();
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

            _dataPoints.AddRange(validData);
            _hasCollectedData = true;
        }

        /// <inheritdoc />
        public DataStructures.QuantizationParameters GetQuantizationParameters()
        {
            if (!_hasCollectedData)
            {
                throw new InvalidOperationException("No statistics have been collected. Call CollectStatistics first.");
            }

            if (_dataPoints.Count == 0)
            {
                throw new InvalidOperationException("No valid data points available after filtering NaN/Inf values.");
            }

            // Calculate percentiles
            var sortedData = _dataPoints.OrderBy(x => x).ToArray();
            int lowerIndex = (int)Math.Ceiling(sortedData.Length * (100 - _percentile) / 100.0);
            int upperIndex = (int)Math.Floor(sortedData.Length * _percentile / 100.0);

            // Ensure indices are valid
            lowerIndex = Math.Max(0, Math.Min(lowerIndex, sortedData.Length - 1));
            upperIndex = Math.Max(0, Math.Min(upperIndex, sortedData.Length - 1));

            float min = sortedData[lowerIndex];
            float max = sortedData[upperIndex];

            // Handle edge case: all values are the same after percentile filtering
            if (Math.Abs(max - min) < float.Epsilon)
            {
                // Fallback to actual min/max
                min = sortedData[0];
                max = sortedData[sortedData.Length - 1];

                if (Math.Abs(max - min) < float.Epsilon)
                {
                    // Still all same values, set a small range
                    min -= 1.0f;
                    max += 1.0f;
                }
            }

            float scale;
            int zeroPoint;

            if (_symmetric)
            {
                // Symmetric quantization
                scale = (float)Math.Max(Math.Abs(min), Math.Abs(max)) / (_quantMax - _quantMin);
                zeroPoint = 0;
            }
            else
            {
                // Asymmetric quantization
                scale = (max - min) / (_quantMax - _quantMin);
                zeroPoint = (int)Math.Round(_quantMin - min / scale);

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
            _dataPoints.Clear();
            _hasCollectedData = false;
        }
    }
}
