using System;
using System.Collections.Generic;
using System.Linq;
using MLFramework.Data;
using MLFramework.NN;
using MLFramework.Quantization.Calibration;
using MLFramework.Quantization.DataStructures;
using RitterFramework.Core.Tensor;

namespace MLFramework.Quantization.PTQ
{
    /// <summary>
    /// Post-training quantization (PTQ) implementation with support for dynamic and static quantization.
    /// </summary>
    public class PTQQuantizer : IQuantizer
    {
        private readonly IModelTraversal _modelTraversal;
        private readonly ICalibrationProcess _calibrationProcess;
        private readonly IDynamicQuantization _dynamicQuantization;
        private readonly IStaticQuantization _staticQuantization;

        private readonly Dictionary<string, bool> _layerFallback;
        private readonly Dictionary<string, QuantizationParameters> _layerQuantizationParams;
        private readonly List<string> _skippedLayers;

        /// <summary>
        /// Creates a new PTQQuantizer instance.
        /// </summary>
        public PTQQuantizer()
            : this(new CalibratorFactory(), new ModelTraversal(), new CalibrationProcess())
        {
        }

        /// <summary>
        /// Creates a new PTQQuantizer instance with custom components.
        /// </summary>
        /// <param name="calibratorFactory">Factory for creating calibrators.</param>
        /// <param name="modelTraversal">Model traversal utility.</param>
        /// <param name="calibrationProcess">Calibration process implementation.</param>
        public PTQQuantizer(
            ICalibratorFactory calibratorFactory,
            IModelTraversal modelTraversal,
            ICalibrationProcess calibrationProcess)
        {
            _modelTraversal = modelTraversal ?? throw new ArgumentNullException(nameof(modelTraversal));
            _calibrationProcess = calibrationProcess ?? throw new ArgumentNullException(nameof(calibrationProcess));

            _dynamicQuantization = new DynamicQuantization(_modelTraversal, calibratorFactory);
            _staticQuantization = new StaticQuantization(_modelTraversal, calibratorFactory, _calibrationProcess);

            _layerFallback = new Dictionary<string, bool>();
            _layerQuantizationParams = new Dictionary<string, QuantizationParameters>();
            _skippedLayers = new List<string>();
        }

        /// <summary>
        /// Quantizes the given model using the provided calibration data and configuration.
        /// </summary>
        /// <param name="model">The model to quantize.</param>
        /// <param name="calibrationData">Data loader providing calibration data.</param>
        /// <param name="config">Quantization configuration.</param>
        /// <returns>The quantized model.</returns>
        public Module Quantize(Module model, DataLoader<object> calibrationData, QuantizationConfig config)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            if (calibrationData == null)
                throw new ArgumentNullException(nameof(calibrationData));
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            config.Validate();

            // Clear previous state
            _layerFallback.Clear();
            _layerQuantizationParams.Clear();
            _skippedLayers.Clear();

            // Perform sensitivity analysis if fallback is enabled
            if (config.FallbackToFP32)
            {
                PerformSensitivityAnalysis(model, calibrationData, config);
            }

            // Determine quantization type (dynamic vs static)
            bool isStaticQuantization = calibrationData != null;

            if (isStaticQuantization)
            {
                return _staticQuantization.Quantize(model, calibrationData, config, _layerFallback, _layerQuantizationParams, _skippedLayers);
            }
            else
            {
                return _dynamicQuantization.Quantize(model, config, _layerFallback, _layerQuantizationParams, _skippedLayers);
            }
        }

        /// <summary>
        /// Enables or disables quantization for a specific layer.
        /// </summary>
        /// <param name="layerName">The name of the layer.</param>
        /// <param name="enabled">True to enable quantization, false to disable (fallback to FP32).</param>
        public void SetPerLayerFallback(string layerName, bool enabled)
        {
            if (string.IsNullOrWhiteSpace(layerName))
                throw new ArgumentException("Layer name cannot be null or whitespace.", nameof(layerName));

            _layerFallback[layerName] = !enabled;
        }

        /// <summary>
        /// Gets the quantization parameters for a specific layer.
        /// </summary>
        /// <param name="layerName">The name of the layer.</param>
        /// <returns>Quantization parameters for the layer.</returns>
        public QuantizationParameters GetLayerQuantizationParameters(string layerName)
        {
            if (string.IsNullOrWhiteSpace(layerName))
                throw new ArgumentException("Layer name cannot be null or whitespace.", nameof(layerName));

            if (_layerQuantizationParams.TryGetValue(layerName, out var parameters))
                return parameters;

            throw new KeyNotFoundException($"No quantization parameters found for layer '{layerName}'.");
        }

        /// <summary>
        /// Gets a list of all layers that were skipped during quantization.
        /// </summary>
        /// <returns>List of layer names that were not quantized.</returns>
        public string[] GetSkippedLayers()
        {
            return _skippedLayers.ToArray();
        }

        /// <summary>
        /// Performs sensitivity analysis to identify layers that should fallback to FP32.
        /// </summary>
        private void PerformSensitivityAnalysis(Module model, DataLoader<object> calibrationData, QuantizationConfig config)
        {
            var sensitivityAnalyzer = new SensitivityAnalysis(_calibrationProcess, _modelTraversal);
            var sensitivityResults = sensitivityAnalyzer.Analyze(model, calibrationData, config);

            foreach (var result in sensitivityResults)
            {
                if (result.AccuracyLoss > config.AccuracyThreshold)
                {
                    _layerFallback[result.LayerName] = true;
                    _skippedLayers.Add(result.LayerName);
                }
            }
        }
    }
}
