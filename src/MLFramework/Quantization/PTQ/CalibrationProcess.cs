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
    /// Implementation of the calibration process for static quantization.
    /// </summary>
    public class CalibrationProcess : ICalibrationProcess
    {
        private readonly ICalibratorFactory _calibratorFactory;
        private readonly IModelTraversal _modelTraversal;

        private readonly Dictionary<string, List<ActivationStatistics>> _activationStats;
        private readonly Dictionary<string, ICalibrator> _layerCalibrators;

        /// <summary>
        /// Creates a new CalibrationProcess instance.
        /// </summary>
        public CalibrationProcess()
            : this(new CalibratorFactory(), new ModelTraversal())
        {
        }

        /// <summary>
        /// Creates a new CalibrationProcess instance with custom components.
        /// </summary>
        /// <param name="calibratorFactory">Factory for creating calibrators.</param>
        /// <param name="modelTraversal">Model traversal utility.</param>
        public CalibrationProcess(ICalibratorFactory calibratorFactory, IModelTraversal modelTraversal)
        {
            _calibratorFactory = calibratorFactory ?? throw new ArgumentNullException(nameof(calibratorFactory));
            _modelTraversal = modelTraversal ?? throw new ArgumentNullException(nameof(modelTraversal));

            _activationStats = new Dictionary<string, List<ActivationStatistics>>();
            _layerCalibrators = new Dictionary<string, ICalibrator>();
        }

        /// <summary>
        /// Runs calibration on the model with the provided data.
        /// </summary>
        /// <param name="model">The model to calibrate.</param>
        /// <param name="dataLoader">Data loader for calibration data.</param>
        /// <param name="config">Quantization configuration.</param>
        /// <returns>Dictionary mapping layer names to quantization parameters.</returns>
        public Dictionary<string, QuantizationParameters> RunCalibration(
            Module model,
            DataLoader<object> dataLoader,
            QuantizationConfig config)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            if (dataLoader == null)
                throw new ArgumentNullException(nameof(dataLoader));
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            Reset();

            // Get quantizable layers
            var quantizableLayers = _modelTraversal.GetQuantizableLayers(model);

            // Initialize calibrators for each layer
            foreach (var layer in quantizableLayers)
            {
                var layerName = _modelTraversal.GetLayerName(layer);
                var calibrator = _calibratorFactory.Create(config.CalibrationMethod);
                _layerCalibrators[layerName] = calibrator;
                _activationStats[layerName] = new List<ActivationStatistics>();
            }

            // Run inference and collect statistics
            int batchCount = 0;
            int maxBatches = (int)Math.Min(dataLoader.NumBatches, int.MaxValue);

            foreach (var batch in dataLoader)
            {
                // Convert batch to tensor if needed
                Tensor input = ConvertToTensor(batch);

                // Run inference with hooks to collect activations
                var activations = RunInferenceWithActivationCollection(model, input);

                // Collect statistics for each layer
                foreach (var kvp in activations)
                {
                    var layerName = kvp.Key;
                    var activation = kvp.Value;

                    if (_activationStats.ContainsKey(layerName))
                    {
                        var stats = CollectActivationStatistics(null, activation);
                        _activationStats[layerName].Add(stats);

                        // Feed data to calibrator
                        float[] data = activation.Data;
                        _layerCalibrators[layerName].CollectStatistics(data);
                    }
                }

                batchCount++;

                // Stop after specified number of batches if configured
                if (batchCount >= maxBatches)
                    break;
            }

            // Compute quantization parameters
            var quantizationParams = new Dictionary<string, QuantizationParameters>();
            foreach (var kvp in _layerCalibrators)
            {
                var layerName = kvp.Key;
                var calibrator = kvp.Value;
                quantizationParams[layerName] = calibrator.GetQuantizationParameters();
            }

            return quantizationParams;
        }

        /// <summary>
        /// Collects activation statistics for each layer during inference.
        /// </summary>
        /// <param name="layer">The layer to collect statistics for.</param>
        /// <param name="activation">The activation tensor.</param>
        /// <returns>Statistics including min, max, and histogram data.</returns>
        public ActivationStatistics CollectActivationStatistics(Module layer, Tensor activation)
        {
            if (activation == null)
                throw new ArgumentNullException(nameof(activation));

            float[] data = activation.Data;
            if (data.Length == 0)
                return new ActivationStatistics { SampleCount = 0 };

            float min = data[0];
            float max = data[0];
            float sum = 0;
            float sumSquares = 0;

            foreach (float val in data)
            {
                if (val < min) min = val;
                if (val > max) max = val;
                sum += val;
                sumSquares += val * val;
            }

            int n = data.Length;
            float mean = sum / n;
            float variance = (sumSquares / n) - (mean * mean);
            float stdDev = MathF.Sqrt(MathF.Max(0, variance));

            // Create histogram (50 bins)
            float range = max - min;
            int numBins = 50;
            var histogram = new float[numBins];
            float binWidth = range / numBins;

            if (binWidth > 0)
            {
                foreach (float val in data)
                {
                    int bin = (int)((val - min) / binWidth);
                    bin = Math.Clamp(bin, 0, numBins - 1);
                    histogram[bin]++;
                }

                // Normalize histogram
                for (int i = 0; i < numBins; i++)
                {
                    histogram[i] /= n;
                }
            }

            return new ActivationStatistics
            {
                Min = min,
                Max = max,
                Mean = mean,
                StdDev = stdDev,
                Histogram = histogram,
                SampleCount = n
            };
        }

        /// <summary>
        /// Resets all calibration statistics.
        /// </summary>
        public void Reset()
        {
            _activationStats.Clear();
            _layerCalibrators.Clear();
        }

        /// <summary>
        /// Runs inference and collects activations from each layer.
        /// </summary>
        private Dictionary<string, Tensor> RunInferenceWithActivationCollection(Module model, Tensor input)
        {
            var activations = new Dictionary<string, Tensor>();
            var layers = _modelTraversal.GetAllLayers(model);

            Tensor current = input;

            foreach (var layer in layers)
            {
                var layerName = _modelTraversal.GetLayerName(layer);

                // Store activation before layer processing
                activations[layerName] = current;

                // Process through layer
                current = layer.Forward(current);
            }

            return activations;
        }

        /// <summary>
        /// Converts batch object to tensor.
        /// </summary>
        private Tensor ConvertToTensor(object batch)
        {
            if (batch is Tensor tensor)
                return tensor;

            // Handle tuple batch format (input, target)
            // This is a simplified check - real implementation would be more robust
            if (batch != null && batch.GetType().IsGenericType &&
                batch.GetType().GetGenericTypeDefinition() == typeof(ValueTuple<,>))
            {
                var inputProperty = batch.GetType().GetProperty("Item1");
                if (inputProperty != null && inputProperty.GetValue(batch) is Tensor inputTensor)
                    return inputTensor;
            }

            // Handle dictionary or other batch formats as needed
            throw new NotSupportedException($"Batch type '{batch?.GetType().Name}' is not supported for calibration.");
        }
    }
}
