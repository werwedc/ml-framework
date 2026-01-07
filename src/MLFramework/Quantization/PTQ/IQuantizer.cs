using MLFramework.Data;
using MLFramework.NN;
using MLFramework.Quantization.DataStructures;

namespace MLFramework.Quantization.PTQ
{
    /// <summary>
    /// Interface for quantizers that apply post-training quantization to models.
    /// </summary>
    public interface IQuantizer
    {
        /// <summary>
        /// Quantizes the given model using the provided calibration data and configuration.
        /// </summary>
        /// <param name="model">The model to quantize.</param>
        /// <param name="calibrationData">Data loader providing calibration data.</param>
        /// <param name="config">Quantization configuration.</param>
        /// <returns>The quantized model.</returns>
        Module Quantize(Module model, DataLoader<object> calibrationData, QuantizationConfig config);

        /// <summary>
        /// Enables or disables quantization for a specific layer.
        /// </summary>
        /// <param name="layerName">The name of the layer.</param>
        /// <param name="enabled">True to enable quantization, false to disable (fallback to FP32).</param>
        void SetPerLayerFallback(string layerName, bool enabled);

        /// <summary>
        /// Gets the quantization parameters for a specific layer.
        /// </summary>
        /// <param name="layerName">The name of the layer.</param>
        /// <returns>Quantization parameters for the layer.</returns>
        QuantizationParameters GetLayerQuantizationParameters(string layerName);

        /// <summary>
        /// Gets a list of all layers that were skipped during quantization.
        /// </summary>
        /// <returns>List of layer names that were not quantized.</returns>
        string[] GetSkippedLayers();
    }
}
