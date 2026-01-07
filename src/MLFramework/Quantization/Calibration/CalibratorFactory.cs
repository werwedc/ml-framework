using MLFramework.Quantization.DataStructures;

namespace MLFramework.Quantization.Calibration
{
    /// <summary>
    /// Factory for creating calibrator instances based on calibration method.
    /// </summary>
    public class CalibratorFactory : ICalibratorFactory
    {
        /// <summary>
        /// Creates a calibrator instance based on the specified calibration method.
        /// </summary>
        /// <param name="method">The calibration method to use.</param>
        /// <returns>An instance of ICalibrator appropriate for the specified method.</returns>
        /// <exception cref="ArgumentException">Thrown when an unknown calibration method is specified.</exception>
        public ICalibrator Create(CalibrationMethod method)
        {
            return method switch
            {
                CalibrationMethod.MinMax => new MinMaxCalibrator(),
                CalibrationMethod.Entropy => new EntropyCalibrator(),
                CalibrationMethod.Percentile => new PercentileCalibrator(),
                CalibrationMethod.MovingAverage => new MovingAverageCalibrator(),
                _ => throw new ArgumentException($"Unknown calibration method: {method}", nameof(method))
            };
        }

        /// <summary>
        /// Creates a calibrator instance with custom configuration.
        /// </summary>
        /// <param name="method">The calibration method to use.</param>
        /// <param name="symmetric">Whether to use symmetric quantization.</param>
        /// <param name="quantMin">Minimum quantized value.</param>
        /// <param name="quantMax">Maximum quantized value.</param>
        /// <returns>An instance of ICalibrator with the specified configuration.</returns>
        /// <exception cref="ArgumentException">Thrown when an unknown calibration method is specified.</exception>
        public ICalibrator Create(CalibrationMethod method, bool symmetric, int quantMin = -128, int quantMax = 127)
        {
            return method switch
            {
                CalibrationMethod.MinMax => new MinMaxCalibrator(symmetric, quantMin, quantMax),
                CalibrationMethod.Entropy => new EntropyCalibrator(symmetric, quantMin, quantMax),
                CalibrationMethod.Percentile => new PercentileCalibrator(99.9f, symmetric, quantMin, quantMax),
                CalibrationMethod.MovingAverage => new MovingAverageCalibrator(100, symmetric, quantMin, quantMax),
                _ => throw new ArgumentException($"Unknown calibration method: {method}", nameof(method))
            };
        }

        /// <summary>
        /// Creates a percentile calibrator with a custom percentile value.
        /// </summary>
        /// <param name="percentile">The percentile value (between 50 and 100).</param>
        /// <param name="symmetric">Whether to use symmetric quantization.</param>
        /// <param name="quantMin">Minimum quantized value.</param>
        /// <param name="quantMax">Maximum quantized value.</param>
        /// <returns>A PercentileCalibrator instance with the specified configuration.</returns>
        public ICalibrator CreatePercentile(float percentile, bool symmetric = false, int quantMin = -128, int quantMax = 127)
        {
            return new PercentileCalibrator(percentile, symmetric, quantMin, quantMax);
        }

        /// <summary>
        /// Creates a moving average calibrator with a custom window size.
        /// </summary>
        /// <param name="windowSize">Number of samples to maintain in the moving window.</param>
        /// <param name="symmetric">Whether to use symmetric quantization.</param>
        /// <param name="quantMin">Minimum quantized value.</param>
        /// <param name="quantMax">Maximum quantized value.</param>
        /// <returns>A MovingAverageCalibrator instance with the specified configuration.</returns>
        public ICalibrator CreateMovingAverage(int windowSize, bool symmetric = false, int quantMin = -128, int quantMax = 127)
        {
            return new MovingAverageCalibrator(windowSize, symmetric, quantMin, quantMax);
        }
    }
}
