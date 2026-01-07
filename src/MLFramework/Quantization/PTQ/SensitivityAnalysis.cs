using System;
using System.Collections.Generic;
using MLFramework.Data;
using MLFramework.NN;
using MLFramework.Quantization.DataStructures;

namespace MLFramework.Quantization.PTQ
{
    /// <summary>
    /// Performs per-layer sensitivity analysis to identify layers sensitive to quantization.
    /// </summary>
    public class SensitivityAnalysis
    {
        private readonly ICalibrationProcess _calibrationProcess;
        private readonly IModelTraversal _modelTraversal;

        /// <summary>
        /// Creates a new SensitivityAnalysis instance.
        /// </summary>
        public SensitivityAnalysis()
            : this(new CalibrationProcess(), new ModelTraversal())
        {
        }

        /// <summary>
        /// Creates a new SensitivityAnalysis instance with custom components.
        /// </summary>
        /// <param name="calibrationProcess">Calibration process implementation.</param>
        /// <param name="modelTraversal">Model traversal utility.</param>
        public SensitivityAnalysis(ICalibrationProcess calibrationProcess, IModelTraversal modelTraversal)
        {
            _calibrationProcess = calibrationProcess ?? throw new ArgumentNullException(nameof(calibrationProcess));
            _modelTraversal = modelTraversal ?? throw new ArgumentNullException(nameof(modelTraversal));
        }

        /// <summary>
        /// Analyzes sensitivity for each quantizable layer.
        /// </summary>
        /// <param name="model">The model to analyze.</param>
        /// <param name="dataLoader">Data loader for analysis data.</param>
        /// <param name="config">Quantization configuration.</param>
        /// <returns>List of sensitivity analysis results for each layer.</returns>
        public List<SensitivityResult> Analyze(
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

            var results = new List<SensitivityResult>();
            var quantizableLayers = _modelTraversal.GetQuantizableLayers(model);

            // Run baseline inference with FP32 model
            float baselineAccuracy = EvaluateModelAccuracy(model, dataLoader);

            // Analyze each layer
            foreach (var layer in quantizableLayers)
            {
                var layerName = _modelTraversal.GetLayerName(layer);

                // Collect layer statistics
                var layerStats = CollectLayerStatistics(layer, dataLoader);

                // Estimate sensitivity based on activation distribution
                float sensitivity = EstimateSensitivity(layerStats);

                // Predict accuracy loss
                float predictedAccuracyLoss = baselineAccuracy * sensitivity;

                var result = new SensitivityResult
                {
                    LayerName = layerName,
                    LayerType = layer.GetType().Name,
                    BaselineAccuracy = baselineAccuracy,
                    PredictedAccuracy = baselineAccuracy - predictedAccuracyLoss,
                    AccuracyLoss = predictedAccuracyLoss,
                    SensitivityScore = sensitivity,
                    MinActivation = layerStats.Min,
                    MaxActivation = layerStats.Max,
                    MeanActivation = layerStats.Mean,
                    StdDevActivation = layerStats.StdDev
                };

                results.Add(result);
            }

            return results;
        }

        /// <summary>
        /// Collects statistics for a specific layer.
        /// </summary>
        private ActivationStatistics CollectLayerStatistics(Module layer, DataLoader<object> dataLoader)
        {
            _calibrationProcess.Reset();

            int batchCount = 0;
            int maxBatches = (int)Math.Min(10, dataLoader.NumBatches); // Limit to 10 batches for speed
            ActivationStatistics aggregatedStats = null;

            foreach (var batch in dataLoader)
            {
                // This is a simplified implementation
                // In a real implementation, we would hook into the layer's forward pass
                // to collect actual activation statistics

                batchCount++;
                if (batchCount >= maxBatches)
                    break;
            }

            // Return dummy stats for now
            return aggregatedStats ?? new ActivationStatistics
            {
                Min = -1.0f,
                Max = 1.0f,
                Mean = 0.0f,
                StdDev = 0.5f,
                SampleCount = 1000
            };
        }

        /// <summary>
        /// Estimates sensitivity based on activation distribution.
        /// </summary>
        private float EstimateSensitivity(ActivationStatistics stats)
        {
            // Sensitivity estimation heuristic:
            // - Larger ranges (high std dev) indicate more sensitivity
            // - Skewed distributions (mean far from 0) may be more sensitive
            // - Layers with very small ranges may not benefit much from quantization

            float range = stats.Max - stats.Min;
            float normalizedMean = Math.Abs(stats.Mean) / (Math.Abs(range) + 1e-6f);
            float normalizedStdDev = stats.StdDev / (Math.Abs(range) + 1e-6f);

            // Sensitivity score ranges from 0 (not sensitive) to 1 (very sensitive)
            float sensitivity = 0.5f;

            // Higher std dev increases sensitivity
            sensitivity += normalizedStdDev * 0.3f;

            // Skewed distributions may increase sensitivity
            sensitivity += normalizedMean * 0.2f;

            return Math.Clamp(sensitivity, 0.0f, 1.0f);
        }

        /// <summary>
        /// Evaluates model accuracy on validation data.
        /// </summary>
        private float EvaluateModelAccuracy(Module model, DataLoader<object> dataLoader)
        {
            // This is a simplified implementation
            // In a real implementation, we would run full inference and compute accuracy
            return 0.95f; // Dummy baseline accuracy
        }

        /// <summary>
        /// Generates a sensitivity report.
        /// </summary>
        public string GenerateSensitivityReport(List<SensitivityResult> results)
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("=== Sensitivity Analysis Report ===");
            report.AppendLine();
            report.AppendLine("Layer Name\t\tType\t\tAccuracy Loss\tSensitivity");
            report.AppendLine("--------------------------------------------------------");

            // Sort by accuracy loss (most sensitive first)
            var sortedResults = results.OrderByDescending(r => r.AccuracyLoss).ToList();

            foreach (var result in sortedResults)
            {
                report.AppendLine($"{result.LayerName,-20}\t{result.LayerType,-10}\t{result.AccuracyLoss:P2}\t{result.SensitivityScore:F2}");
            }

            report.AppendLine();
            report.AppendLine($"Total layers analyzed: {results.Count}");
            report.AppendLine($"High sensitivity layers (accuracy loss > 1%): {results.Count(r => r.AccuracyLoss > 0.01f)}");
            report.AppendLine($"Medium sensitivity layers (0.1% < accuracy loss <= 1%): {results.Count(r => r.AccuracyLoss > 0.001f && r.AccuracyLoss <= 0.01f)}");
            report.AppendLine($"Low sensitivity layers (accuracy loss <= 0.1%): {results.Count(r => r.AccuracyLoss <= 0.001f)}");

            return report.ToString();
        }
    }

    /// <summary>
    /// Result of sensitivity analysis for a single layer.
    /// </summary>
    public class SensitivityResult
    {
        /// <summary>
        /// Name of the layer.
        /// </summary>
        public string LayerName { get; set; }

        /// <summary>
        /// Type of the layer.
        /// </summary>
        public string LayerType { get; set; }

        /// <summary>
        /// Baseline accuracy with FP32 model.
        /// </summary>
        public float BaselineAccuracy { get; set; }

        /// <summary>
        /// Predicted accuracy after quantization.
        /// </summary>
        public float PredictedAccuracy { get; set; }

        /// <summary>
        /// Predicted accuracy loss from quantization.
        /// </summary>
        public float AccuracyLoss { get; set; }

        /// <summary>
        /// Sensitivity score (0 = not sensitive, 1 = very sensitive).
        /// </summary>
        public float SensitivityScore { get; set; }

        /// <summary>
        /// Minimum activation value.
        /// </summary>
        public float MinActivation { get; set; }

        /// <summary>
        /// Maximum activation value.
        /// </summary>
        public float MaxActivation { get; set; }

        /// <summary>
        /// Mean activation value.
        /// </summary>
        public float MeanActivation { get; set; }

        /// <summary>
        /// Standard deviation of activation values.
        /// </summary>
        public float StdDevActivation { get; set; }
    }
}
