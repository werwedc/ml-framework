using MLFramework.Models;
using MLFramework.Data;
using MLFramework.Quantization.PTQ;
using MLFramework.Quantization.DataStructures;
using MLFramework.Quantization.Evaluation;

namespace MLFramework.Examples.Quantization
{
    /// <summary>
    /// Sensitivity Analysis Example demonstrates how to perform per-layer sensitivity analysis
    /// to identify which layers are most sensitive to quantization.
    /// This example shows how to run sensitivity analysis, generate reports, and apply automatic fallback.
    /// </summary>
    public class SensitivityAnalysisExample
    {
        /// <summary>
        /// Run the sensitivity analysis example.
        /// </summary>
        public static async Task Run()
        {
            Console.WriteLine("=== Sensitivity Analysis Example ===\n");

            // Step 1: Load model and data
            Console.WriteLine("Step 1: Loading model and test data...");
            var model = await LoadModel();
            var testData = await PrepareTestData();
            var calibrationData = await PrepareCalibrationData();
            Console.WriteLine($"  Model: {model.GetType().Name}");
            Console.WriteLine($"  Test samples: {testData.GetSampleCount():,}\n");

            // Step 2: Evaluate baseline model
            Console.WriteLine("Step 2: Evaluating baseline FP32 model...");
            var baselineAccuracy = await EvaluateModel(model, testData);
            Console.WriteLine($"  Baseline accuracy: {baselineAccuracy:F4}\n");

            // Step 3: Run sensitivity analysis
            Console.WriteLine("Step 3: Running per-layer sensitivity analysis...");
            var sensitivityResults = await RunSensitivityAnalysis(
                model,
                calibrationData,
                testData,
                baselineAccuracy
            );
            Console.WriteLine($"  Analyzed {sensitivityResults.Count} layers\n");

            // Step 4: Display sensitivity results
            Console.WriteLine("Step 4: Sensitivity Analysis Results:");
            Console.WriteLine($"  Layer                    | Accuracy Drop | Sensitive | Recommendation");
            Console.WriteLine(new string('-', 85));
            foreach (var result in sensitivityResults)
            {
                var sensitive = result.IsSensitive ? "Yes" : "No";
                var action = result.RecommendedAction;
                var drop = $"{result.AccuracyImpact:F4}";
                Console.WriteLine($"  {result.LayerName,-24} | {drop,13} | {sensitive,9} | {action}");
            }

            // Step 5: Generate sensitivity report
            Console.WriteLine("\nStep 5: Generating sensitivity report...");
            var report = GenerateSensitivityReport(sensitivityResults, baselineAccuracy);
            Console.WriteLine($"  Sensitive layers: {report.SensitiveLayerCount}");
            Console.WriteLine($"  Safe layers: {report.SafeLayerCount}");
            Console.WriteLine($"  Estimated quantized accuracy: {report.EstimatedQuantizedAccuracy:F4}\n");

            // Step 6: Apply automatic fallback
            Console.WriteLine("Step 6: Applying automatic FP32 fallback for sensitive layers...");
            var config = CreateConfigWithFallback(sensitivityResults);
            PrintFallbackConfig(config);

            // Step 7: Quantize with fallback configuration
            Console.WriteLine("\nStep 7: Quantizing with FP32 fallback...");
            var quantizedModel = await ApplyQuantizationWithFallback(
                model,
                calibrationData,
                config
            );
            Console.WriteLine("  Quantization complete!\n");

            // Step 8: Evaluate final model
            Console.WriteLine("Step 8: Evaluating final quantized model...");
            var finalAccuracy = await EvaluateModel(quantizedModel, testData);
            Console.WriteLine($"  Final accuracy: {finalAccuracy:F4}");
            Console.WriteLine($"  Accuracy drop: {baselineAccuracy - finalAccuracy:F4}");
            Console.WriteLine($"  Drop vs estimate: {Math.Abs(report.EstimatedQuantizedAccuracy - finalAccuracy):F4}");

            // Step 9: Save sensitivity report
            Console.WriteLine("\nStep 9: Saving sensitivity report...");
            await SaveSensitivityReport(report, "sensitivity_report.json");
            Console.WriteLine("  Report saved to: sensitivity_report.json");

            Console.WriteLine("\n=== Sensitivity Analysis Example Complete ===");
        }

        /// <summary>
        /// Run per-layer sensitivity analysis.
        /// </summary>
        private static async Task<List<SensitivityAnalysisResult>> RunSensitivityAnalysis(
            IModel model,
            IDataLoader calibrationData,
            IDataLoader testData,
            float baselineAccuracy)
        {
            var results = new List<SensitivityAnalysisResult>();
            var quantizableLayers = model.GetQuantizableLayerNames();
            var threshold = 0.01f; // 1% accuracy drop threshold

            Console.WriteLine("  Analyzing layers...");
            foreach (var layerName in quantizableLayers)
            {
                Console.Write($"    {layerName}... ");

                // Quantize only this layer
                var config = CreateSingleLayerConfig(layerName);
                var quantizedModel = await QuantizeSingleLayer(model, calibrationData, config);

                // Evaluate
                var accuracy = await EvaluateModel(quantizedModel, testData);
                var accuracyDrop = baselineAccuracy - accuracy;

                // Determine if sensitive
                var isSensitive = accuracyDrop > threshold;

                results.Add(new SensitivityAnalysisResult
                {
                    LayerName = layerName,
                    AccuracyImpact = accuracyDrop,
                    IsSensitive = isSensitive,
                    RecommendedAction = isSensitive ? "Fallback to FP32" : "Quantize to Int8"
                });

                Console.WriteLine($"{accuracyDrop:F4} ({(isSensitive ? "SENSITIVE" : "OK")})");
            }

            // Sort by accuracy impact (descending)
            results = results.OrderByDescending(r => r.AccuracyImpact).ToList();

            return results;
        }

        /// <summary>
        /// Create configuration for quantizing a single layer.
        /// </summary>
        private static QuantizationConfig CreateSingleLayerConfig(string layerName)
        {
            var config = new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerTensorSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorAsymmetric,
                CalibrationMethod = CalibrationMethod.Entropy,
                CalibrationBatchSize = 32,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = false
            };

            // Enable quantization only for the specified layer
            config.EnableLayerQuantization(layerName, enabled: true);

            return config;
        }

        /// <summary>
        /// Quantize a single layer for sensitivity analysis.
        /// </summary>
        private static async Task<IModel> QuantizeSingleLayer(
            IModel model,
            IDataLoader calibrationData,
            QuantizationConfig config)
        {
            var quantizer = new PTQQuantizer();
            var quantizedModel = await quantizer.QuantizeAsync(model, calibrationData, config);
            return quantizedModel;
        }

        /// <summary>
        /// Generate sensitivity analysis report.
        /// </summary>
        private static SensitivityReport GenerateSensitivityReport(
            List<SensitivityAnalysisResult> results,
            float baselineAccuracy)
        {
            var sensitiveLayers = results.Where(r => r.IsSensitive).ToList();
            var safeLayers = results.Where(r => !r.IsSensitive).ToList();

            // Estimate quantized accuracy based on analysis
            var totalDrop = safeLayers.Sum(r => r.AccuracyImpact);
            var estimatedAccuracy = baselineAccuracy - totalDrop;

            var report = new SensitivityReport
            {
                BaselineAccuracy = baselineAccuracy,
                SensitiveLayerCount = sensitiveLayers.Count,
                SafeLayerCount = safeLayers.Count,
                EstimatedQuantizedAccuracy = estimatedAccuracy,
                LayerResults = results
            };

            return report;
        }

        /// <summary>
        /// Create quantization configuration with fallback for sensitive layers.
        /// </summary>
        private static QuantizationConfig CreateConfigWithFallback(
            List<SensitivityAnalysisResult> sensitivityResults)
        {
            var config = new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerTensorSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorAsymmetric,
                CalibrationMethod = CalibrationMethod.Entropy,
                CalibrationBatchSize = 32,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = true
            };

            // Configure per-layer settings based on sensitivity
            foreach (var result in sensitivityResults)
            {
                var layerConfig = new LayerQuantizationConfig
                {
                    Enabled = !result.IsSensitive,
                    QuantizationType = QuantizationType.Int8,
                    QuantizationMode = QuantizationMode.PerTensorSymmetric
                };

                config.SetLayerConfiguration(result.LayerName, layerConfig);
            }

            return config;
        }

        /// <summary>
        /// Print fallback configuration summary.
        /// </summary>
        private static void PrintFallbackConfig(QuantizationConfig config)
        {
            var stats = config.GetConfigurationStatistics();
            Console.WriteLine($"  Total layers: {stats.TotalLayers}");
            Console.WriteLine($"  Quantized layers: {stats.QuantizedLayers}");
            Console.WriteLine($"  FP32 fallback layers: {stats.FP32Layers}");
            Console.WriteLine($"  Fallback ratio: {stats.FP32Layers * 100.0 / stats.TotalLayers:F1}%");
        }

        /// <summary>
        /// Apply quantization with automatic fallback.
        /// </summary>
        private static async Task<IModel> ApplyQuantizationWithFallback(
            IModel model,
            IDataLoader calibrationData,
            QuantizationConfig config)
        {
            var quantizer = new PTQQuantizer();
            var quantizedModel = await quantizer.QuantizeAsync(model, calibrationData, config);
            return quantizedModel;
        }

        /// <summary>
        /// Save sensitivity report to file.
        /// </summary>
        private static async Task SaveSensitivityReport(SensitivityReport report, string filePath)
        {
            var json = System.Text.Json.JsonSerializer.Serialize(report, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true
            });
            await File.WriteAllTextAsync(filePath, json);
        }

        /// <summary>
        /// Load model for analysis.
        /// </summary>
        private static async Task<IModel> LoadModel()
        {
            var model = CreateSampleModel();
            return model;
        }

        /// <summary>
        /// Prepare test data.
        /// </summary>
        private static async Task<IDataLoader> PrepareTestData()
        {
            var testData = CreateSampleTestData();
            return testData;
        }

        /// <summary>
        /// Prepare calibration data.
        /// </summary>
        private static async Task<IDataLoader> PrepareCalibrationData()
        {
            var calibrationData = CreateSampleCalibrationData(100);
            return calibrationData;
        }

        /// <summary>
        /// Evaluate model on test data.
        /// </summary>
        private static async Task<float> EvaluateModel(IModel model, IDataLoader testData)
        {
            var evaluator = new ModelEvaluator();
            var top1Accuracy = new TopKAccuracy(k: 1);
            var accuracy = await evaluator.EvaluateAsync(model, testData, top1Accuracy);
            return accuracy;
        }

        #region Supporting Data Structures

        /// <summary>
        /// Sensitivity analysis result for a single layer.
        /// </summary>
        private class SensitivityAnalysisResult
        {
            public string LayerName { get; set; } = string.Empty;
            public float AccuracyImpact { get; set; }
            public bool IsSensitive { get; set; }
            public string RecommendedAction { get; set; } = string.Empty;
        }

        /// <summary>
        /// Complete sensitivity analysis report.
        /// </summary>
        private class SensitivityReport
        {
            public float BaselineAccuracy { get; set; }
            public int SensitiveLayerCount { get; set; }
            public int SafeLayerCount { get; set; }
            public float EstimatedQuantizedAccuracy { get; set; }
            public List<SensitivityAnalysisResult> LayerResults { get; set; } = new();
        }

        #endregion

        #region Sample Implementation (Replace with real code in production)

        private static IModel CreateSampleModel()
        {
            // Placeholder: Create a sample model
            throw new NotImplementedException("Replace with actual model creation code");
        }

        private static IDataLoader CreateSampleTestData()
        {
            // Placeholder: Create sample test data
            throw new NotImplementedException("Replace with actual data loading code");
        }

        private static IDataLoader CreateSampleCalibrationData(int sampleCount)
        {
            // Placeholder: Create sample calibration data
            throw new NotImplementedException("Replace with actual data loading code");
        }

        #endregion

        /// <summary>
        /// Main entry point for the example.
        /// </summary>
        public static async Task Main(string[] args)
        {
            await Run();
        }
    }
}
