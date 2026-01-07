using System;

namespace MLFramework.Quantization.Calibration
{
    /// <summary>
    /// Interface for calibrators that determine quantization parameters from data statistics.
    /// </summary>
    public interface ICalibrator
    {
        /// <summary>
        /// Collects statistics from the provided data.
        /// </summary>
        /// <param name="data">Array of float values to collect statistics from.</param>
        void CollectStatistics(float[] data);

        /// <summary>
        /// Gets the quantization parameters based on collected statistics.
        /// </summary>
        /// <returns>QuantizationParameters with scale and zero-point.</returns>
        /// <exception cref="InvalidOperationException">Thrown when no statistics have been collected.</exception>
        Quantization.DataStructures.QuantizationParameters GetQuantizationParameters();

        /// <summary>
        /// Resets the calibrator state, clearing all collected statistics.
        /// </summary>
        void Reset();
    }
}
