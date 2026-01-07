using System.Collections.Generic;
using MLFramework.NN;
using MLFramework.Quantization.DataStructures;

namespace MLFramework.Quantization.PTQ
{
    /// <summary>
    /// Interface for dynamic quantization (weights quantized to Int8, activations in FP32).
    /// </summary>
    public interface IDynamicQuantization
    {
        /// <summary>
        /// Applies dynamic quantization to the model.
        /// </summary>
        /// <param name="model">The model to quantize.</param>
        /// <param name="config">Quantization configuration.</param>
        /// <param name="layerFallback">Dictionary indicating which layers should fallback to FP32.</param>
        /// <param name="layerParams">Dictionary to store quantization parameters per layer.</param>
        /// <param name="skippedLayers">List to store layers that were skipped.</param>
        /// <returns>The quantized model.</returns>
        Module Quantize(
            Module model,
            QuantizationConfig config,
            Dictionary<string, bool> layerFallback,
            Dictionary<string, QuantizationParameters> layerParams,
            List<string> skippedLayers);
    }
}
