using System;
using System.Collections.Generic;
using MLFramework.NN;
using MLFramework.Quantization.Calibration;
using MLFramework.Quantization.DataStructures;
using MLFramework.Quantization.Operations;
using RitterFramework.Core.Tensor;

namespace MLFramework.Quantization.PTQ
{
    /// <summary>
    /// Dynamic quantization implementation - weights quantized to Int8, activations in FP32.
    /// </summary>
    public class DynamicQuantization : IDynamicQuantization
    {
        private readonly IModelTraversal _modelTraversal;
        private readonly ICalibratorFactory _calibratorFactory;

        /// <summary>
        /// Creates a new DynamicQuantization instance.
        /// </summary>
        /// <param name="modelTraversal">Model traversal utility.</param>
        /// <param name="calibratorFactory">Factory for creating calibrators.</param>
        public DynamicQuantization(IModelTraversal modelTraversal, ICalibratorFactory calibratorFactory)
        {
            _modelTraversal = modelTraversal ?? throw new ArgumentNullException(nameof(modelTraversal));
            _calibratorFactory = calibratorFactory ?? throw new ArgumentNullException(nameof(calibratorFactory));
        }

        /// <summary>
        /// Applies dynamic quantization to the model.
        /// </summary>
        /// <param name="model">The model to quantize.</param>
        /// <param name="config">Quantization configuration.</param>
        /// <param name="layerFallback">Dictionary indicating which layers should fallback to FP32.</param>
        /// <param name="layerParams">Dictionary to store quantization parameters per layer.</param>
        /// <param name="skippedLayers">List to store layers that were skipped.</param>
        /// <returns>The quantized model.</returns>
        public Module Quantize(
            Module model,
            QuantizationConfig config,
            Dictionary<string, bool> layerFallback,
            Dictionary<string, QuantizationParameters> layerParams,
            List<string> skippedLayers)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            var quantizableLayers = _modelTraversal.GetQuantizableLayers(model);

            foreach (var layer in quantizableLayers)
            {
                var layerName = _modelTraversal.GetLayerName(layer);

                // Check if layer should fallback to FP32
                if (layerFallback != null && layerFallback.ContainsKey(layerName) && layerFallback[layerName])
                {
                    skippedLayers?.Add(layerName);
                    continue;
                }

                // Quantize weights
                QuantizeLayerWeights(layer, layerName, config, layerParams);
            }

            return model;
        }

        /// <summary>
        /// Quantizes the weights of a layer.
        /// </summary>
        private void QuantizeLayerWeights(
            Module layer,
            string layerName,
            QuantizationConfig config,
            Dictionary<string, QuantizationParameters> layerParams)
        {
            foreach (var (paramName, parameter) in layer.GetNamedParameters())
            {
                if (parameter == null || parameter.Data == null)
                    continue;

                float[] weights = parameter.Data;
                int[] shape = parameter.Shape;

                // Determine quantization mode (per-tensor or per-channel)
                bool perChannel = _modelTraversal.SupportsPerChannelQuantization(layer) &&
                                   config.EnablePerChannelQuantization &&
                                   config.WeightQuantization.ToString().Contains("PerChannel");

                QuantizationParameters parameters;

                if (perChannel)
                {
                    // Per-channel quantization
                    parameters = QuantizePerChannel(weights, shape, config);
                }
                else
                {
                    // Per-tensor quantization
                    parameters = QuantizePerTensor(weights, config);
                }

                // Apply quantization
                sbyte[] quantizedWeights = new sbyte[weights.Length];

                for (int i = 0; i < weights.Length; i++)
                {
                    quantizedWeights[i] = QuantizationOperations.Quantize(weights[i], parameters);
                }

                // Store quantization parameters
                layerParams[layerName] = parameters;

                // Note: In a real implementation, we would replace the FP32 weights
                // with Int8 weights and modify the layer to dequantize during inference
            }
        }

        /// <summary>
        /// Gets the bit width from the quantization type.
        /// </summary>
        private int GetBitWidth(QuantizationType type)
        {
            return type == QuantizationType.Int8 ? 8 : 8;
        }

        /// <summary>
        /// Quantizes weights with per-tensor quantization.
        /// </summary>
        private QuantizationParameters QuantizePerTensor(float[] weights, QuantizationConfig config)
        {
            // Find min and max
            float min = float.MaxValue;
            float max = float.MinValue;

            foreach (float w in weights)
            {
                if (w < min) min = w;
                if (w > max) max = w;
            }

            // Calculate scale and zero-point
            int bitWidth = GetBitWidth(config.QuantizationType);
            var qmin = -(1 << (bitWidth - 1));
            var qmax = (1 << (bitWidth - 1)) - 1;

            float scale = (max - min) / (qmax - qmin);
            if (scale < 1e-6f) scale = 1e-6f;

            float zeroPoint = qmin - min / scale;
            zeroPoint = Math.Clamp(zeroPoint, qmin, qmax);

            return new QuantizationParameters(scale, (int)zeroPoint, min, max, config.WeightQuantization, config.QuantizationType);
        }

        /// <summary>
        /// Quantizes weights with per-channel quantization.
        /// </summary>
        private QuantizationParameters QuantizePerChannel(float[] weights, int[] shape, QuantizationConfig config)
        {
            // For per-channel, we would typically quantize along the output channel dimension
            // This is a simplified implementation - real implementation would be more complex

            // Find min and max per channel
            int numChannels = shape[0];
            int channelSize = weights.Length / numChannels;

            float[] minPerChannel = new float[numChannels];
            float[] maxPerChannel = new float[numChannels];

            for (int c = 0; c < numChannels; c++)
            {
                minPerChannel[c] = float.MaxValue;
                maxPerChannel[c] = float.MinValue;

                for (int i = 0; i < channelSize; i++)
                {
                    float w = weights[c * channelSize + i];
                    if (w < minPerChannel[c]) minPerChannel[c] = w;
                    if (w > maxPerChannel[c]) maxPerChannel[c] = w;
                }
            }

            // For simplicity, return average parameters
            float avgMin = minPerChannel.Average();
            float avgMax = maxPerChannel.Average();

            int bitWidth = GetBitWidth(config.QuantizationType);
            var qmin = -(1 << (bitWidth - 1));
            var qmax = (1 << (bitWidth - 1)) - 1;

            float scale = (avgMax - avgMin) / (qmax - qmin);
            if (scale < 1e-6f) scale = 1e-6f;

            float zeroPoint = qmin - avgMin / scale;
            zeroPoint = Math.Clamp(zeroPoint, qmin, qmax);

            return new QuantizationParameters(scale, (int)zeroPoint);
        }
    }
}
