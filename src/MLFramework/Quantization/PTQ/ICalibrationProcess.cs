using System;
using System.Collections.Generic;
using MLFramework.Data;
using MLFramework.NN;
using MLFramework.Quantization.DataStructures;
using RitterFramework.Core.Tensor;

namespace MLFramework.Quantization.PTQ
{
    /// <summary>
    /// Interface for calibration process during static quantization.
    /// </summary>
    public interface ICalibrationProcess
    {
        /// <summary>
        /// Runs calibration on the model with the provided data.
        /// </summary>
        /// <param name="model">The model to calibrate.</param>
        /// <param name="dataLoader">Data loader for calibration data.</param>
        /// <param name="config">Quantization configuration.</param>
        /// <returns>Dictionary mapping layer names to quantization parameters.</returns>
        Dictionary<string, QuantizationParameters> RunCalibration(
            Module model,
            DataLoader<object> dataLoader,
            QuantizationConfig config);

        /// <summary>
        /// Collects activation statistics for each layer during inference.
        /// </summary>
        /// <param name="layer">The layer to collect statistics for.</param>
        /// <param name="activation">The activation tensor.</param>
        /// <returns>Statistics including min, max, and histogram data.</returns>
        ActivationStatistics CollectActivationStatistics(Module layer, Tensor activation);

        /// <summary>
        /// Resets all calibration statistics.
        /// </summary>
        void Reset();
    }

    /// <summary>
    /// Statistics collected during calibration.
    /// </summary>
    public class ActivationStatistics
    {
        /// <summary>
        /// Minimum activation value.
        /// </summary>
        public float Min { get; set; }

        /// <summary>
        /// Maximum activation value.
        /// </summary>
        public float Max { get; set; }

        /// <summary>
        /// Mean activation value.
        /// </summary>
        public float Mean { get; set; }

        /// <summary>
        /// Standard deviation of activation values.
        /// </summary>
        public float StdDev { get; set; }

        /// <summary>
        /// Histogram of activation values for entropy-based calibration.
        /// </summary>
        public float[] Histogram { get; set; }

        /// <summary>
        /// Number of samples collected.
        /// </summary>
        public int SampleCount { get; set; }
    }
}
