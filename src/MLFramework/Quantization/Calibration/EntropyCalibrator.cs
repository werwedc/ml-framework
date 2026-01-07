using System;
using System.Linq;

namespace MLFramework.Quantization.Calibration
{
    /// <summary>
    /// Calibrator that uses KL divergence to find optimal quantization parameters.
    /// Computes a histogram of activation values and finds the cut-off point that minimizes KL divergence.
    /// Better accuracy for asymmetric distributions compared to min/max calibration.
    /// </summary>
    public class EntropyCalibrator : ICalibrator
    {
        private const int HistogramBins = 2048;
        private readonly bool _symmetric;
        private readonly int _quantMin;
        private readonly int _quantMax;
        private System.Collections.Generic.List<float> _dataPoints;
        private bool _hasCollectedData;

        /// <summary>
        /// Initializes a new instance of the EntropyCalibrator class.
        /// </summary>
        /// <param name="symmetric">If true, uses symmetric quantization (zero-point = 0).</param>
        /// <param name="quantMin">Minimum quantized value (default: -128 for Int8).</param>
        /// <param name="quantMax">Maximum quantized value (default: 127 for Int8).</param>
        public EntropyCalibrator(bool symmetric = false, int quantMin = -128, int quantMax = 127)
        {
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

            // Find min/max
            float min = _dataPoints.Min();
            float max = _dataPoints.Max();

            // Handle edge case: all values are the same
            if (Math.Abs(max - min) < float.Epsilon)
            {
                min -= 1.0f;
                max += 1.0f;
            }

            // For asymmetric distributions, use KL divergence to find optimal cut-off
            if (!_symmetric && _dataPoints.Count > 100)
            {
                float optimalMax = FindOptimalCutoff(min, max);
                return ComputeParameters(min, optimalMax);
            }

            // For symmetric mode or small datasets, use simple min/max
            return ComputeParameters(min, max);
        }

        /// <summary>
        /// Finds the optimal cut-off point using KL divergence.
        /// </summary>
        private float FindOptimalCutoff(float min, float max)
        {
            // Build histogram
            var histogram = BuildHistogram(min, max);
            float[] distributionP = NormalizeHistogram(histogram);

            float bestKlDivergence = float.MaxValue;
            float bestCutoff = max;

            // Try different cut-off points (we want to cut off high values that are outliers)
            // We search from the right side, removing bins and computing KL divergence
            for (int binsToRemove = 0; binsToRemove <= histogram.Length / 4; binsToRemove++)
            {
                int cutoffBin = histogram.Length - 1 - binsToRemove;
                if (cutoffBin < histogram.Length / 2) break; // Don't cut more than half

                // Create truncated distribution Q
                int truncatedLength = cutoffBin + 1;
                float[] distributionQ = new float[truncatedLength];
                Array.Copy(distributionP, 0, distributionQ, 0, truncatedLength);

                // Normalize Q
                float sumQ = distributionQ.Sum();
                if (sumQ <= 0) break;

                for (int i = 0; i < distributionQ.Length; i++)
                {
                    distributionQ[i] /= sumQ;
                }

                // Compute KL divergence: D(P||Q) = sum(P[i] * log(P[i] / Q[i]))
                float klDivergence = 0.0f;
                for (int i = 0; i < distributionQ.Length; i++)
                {
                    if (distributionP[i] > 1e-10f && distributionQ[i] > 1e-10f)
                    {
                        klDivergence += distributionP[i] * (float)Math.Log(distributionP[i] / distributionQ[i]);
                    }
                }

                if (klDivergence < bestKlDivergence)
                {
                    bestKlDivergence = klDivergence;
                    bestCutoff = min + (cutoffBin + 0.5f) * (max - min) / histogram.Length;
                }
            }

            return bestCutoff;
        }

        /// <summary>
        /// Builds a histogram of the data points.
        /// </summary>
        private int[] BuildHistogram(float min, float max)
        {
            int[] histogram = new int[HistogramBins];
            float range = max - min;

            foreach (float value in _dataPoints)
            {
                int bin = (int)((value - min) / range * HistogramBins);
                bin = Math.Max(0, Math.Min(bin, HistogramBins - 1));
                histogram[bin]++;
            }

            return histogram;
        }

        /// <summary>
        /// Normalizes histogram to create a probability distribution.
        /// </summary>
        private float[] NormalizeHistogram(int[] histogram)
        {
            float[] distribution = new float[histogram.Length];
            float sum = histogram.Sum();

            for (int i = 0; i < histogram.Length; i++)
            {
                distribution[i] = histogram[i] / sum;
            }

            return distribution;
        }

        /// <summary>
        /// Computes quantization parameters from min/max values.
        /// </summary>
        private DataStructures.QuantizationParameters ComputeParameters(float min, float max)
        {
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
