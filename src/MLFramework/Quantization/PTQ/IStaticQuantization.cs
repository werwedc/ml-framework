using System.Collections.Generic;
using MLFramework.Data;
using MLFramework.NN;
using MLFramework.Quantization.DataStructures;

namespace MLFramework.Quantization.PTQ
{
    /// <summary>
    /// Interface for static quantization (weights and activations both quantized to Int8).
    /// </summary>
    public interface IStaticQuantization
    {
        /// <summary>
        /// Applies static quantization to the model.
        /// </summary>
        /// <param name="model">The model to quantize.</param>
        /// <param name="dataLoader">Data loader for calibration.</param>
        /// <param name="config">Quantization configuration.</param>
        /// <param name="layerFallback">Dictionary indicating which layers should fallback to FP32.</param>
        /// <param name="layerParams">Dictionary to store quantization parameters per layer.</param>
        /// <param name="skippedLayers">List to store layers that were skipped.</param>
        /// <returns>The quantized model.</returns>
        Module Quantize(
            Module model,
            DataLoader<object> dataLoader,
            QuantizationConfig config,
            Dictionary<string, bool> layerFallback,
            Dictionary<string, QuantizationParameters> layerParams,
            List<string> skippedLayers);
    }
}
