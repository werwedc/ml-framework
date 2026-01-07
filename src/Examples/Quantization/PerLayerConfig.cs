using MLFramework.Models;
using MLFramework.Data;
using MLFramework.Quantization.PTQ;
using MLFramework.Quantization.DataStructures;
using MLFramework.Quantization.Evaluation;

namespace MLFramework.Examples.Quantization
{
    /// <summary>
    /// Per-Layer Configuration Example demonstrates how to configure quantization on a per-layer basis.
    /// This example shows how to enable/disable quantization for specific layers, use mixed precision,
    /// and adjust configuration based on layer sensitivity.
    /// </summary>
    public class PerLayerConfig
    {
        /// <summary>
        /// Run the per-layer configuration example.
        /// </summary>
        public static async Task Run()
        {
            Console.WriteLine("=== Per-Layer Configuration Example ===\n");

            // Step 1: Load model
            Console.WriteLine("Step 1: Loading model...");
            var model = await LoadModel();
            Console.WriteLine($"  Model: {model.GetType().Name}");
            Console.WriteLine($"  Total layers: {model.GetLayerCount()}\n");

            // Step 2: Identify quantizable layers
            Console.WriteLine("Step 2: Identifying quantizable layers...");
            var quantizableLayers = model.GetQuantizableLayerNames();
            Console.WriteLine($"  Quantizable layers: {quantizableLayers.Count}");
            Console.WriteLine($"  Layers to quantize: {string.Join(", ", quantizableLayers.Take(5))}...\n");

            // Step 3: Configure per-layer quantization
            Console.WriteLine("Step 3: Configuring per-layer quantization...");
            var config = CreatePerLayerConfig(model);
            PrintConfiguration(config);

            // Step 4: Apply quantization with per-layer config
            Console.WriteLine("Step 4: Applying quantization with per-layer configuration...");
            var calibrationData = await PrepareCalibrationData();
            var quantizedModel = await ApplyPerLayerQuantization(model, calibrationData, config);
            Console.WriteLine($"  Quantization complete!\n");

            // Step 5: Evaluate mixed-precision model
            Console.WriteLine("Step 5: Evaluating mixed-precision model...");
            var testData = await PrepareTestData();
            var accuracy = await EvaluateModel(quantizedModel, testData);
            Console.WriteLine($"  Model accuracy: {accuracy:F4}\n");

            // Step 6: Analyze quantization distribution
            Console.WriteLine("Step 6: Analyzing quantization distribution...");
            AnalyzeQuantizationDistribution(quantizedModel);

            Console.WriteLine("\n=== Per-Layer Configuration Example Complete ===");
        }

        /// <summary>
        /// Create per-layer quantization configuration.
        /// </summary>
        private static QuantizationConfig CreatePerLayerConfig(IModel model)
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

            // Configure per-layer settings
            var layerNames = model.GetQuantizableLayerNames();

            Console.WriteLine("  Per-Layer Configuration:");
            Console.WriteLine($"  Layer                    | Quantized | Precision | Reason");
            Console.WriteLine(new string('-', 70));

            foreach (var layerName in layerNames)
            {
                bool shouldQuantize = DetermineLayerQuantization(layerName);
                QuantizationType precision = DetermineLayerPrecision(layerName);

                // Set per-layer configuration
                config.SetLayerConfiguration(layerName, new LayerQuantizationConfig
                {
                    Enabled = shouldQuantize,
                    QuantizationType = precision,
                    QuantizationMode = shouldQuantize ? QuantizationMode.PerTensorSymmetric : QuantizationMode.PerTensorSymmetric
                });

                var reason = GetConfigurationReason(layerName, shouldQuantize, precision);
                Console.WriteLine($"  {layerName,-24} | {shouldQuantize,9} | {precision,9} | {reason}");
            }

            return config;
        }

        /// <summary>
        /// Determine if a layer should be quantized.
        /// </summary>
        private static bool DetermineLayerQuantization(string layerName)
        {
            // Example logic: Skip certain layer types
            if (layerName.Contains("Softmax") || layerName.Contains("Sigmoid"))
            {
                return false; // Keep activation functions in FP32
            }

            if (layerName.Contains("BatchNorm"))
            {
                return false; // Batch normalization often works better in FP32
            }

            if (layerName.Contains("Embedding"))
            {
                return false; // Embedding layers typically stay in FP32
            }

            // Quantize linear and conv layers
            if (layerName.Contains("Linear") || layerName.Contains("Conv"))
            {
                return true;
            }

            // Default: keep in FP32
            return false;
        }

        /// <summary>
        /// Determine precision for a specific layer.
        /// </summary>
        private static QuantizationType DetermineLayerPrecision(string layerName)
        {
            // Example logic: Use mixed precision
            // Early layers might need higher precision
            if (layerName.Contains("Conv") && layerName.Contains("1"))
            {
                return QuantizationType.Int16; // Use Int16 for early conv layers
            }

            // First linear layer after conv layers
            if (layerName.Contains("Linear") && layerName.Contains("0"))
            {
                return QuantizationType.Int16; // Use Int16 for first linear layer
            }

            // Default to Int8
            return QuantizationType.Int8;
        }

        /// <summary>
        /// Get reason for configuration choice.
        /// </summary>
        private static string GetConfigurationReason(string layerName, bool quantized, QuantizationType precision)
        {
            if (!quantized)
            {
                if (layerName.Contains("Softmax"))
                    return "Output stability";
                if (layerName.Contains("Sigmoid"))
                    return "Gradient flow";
                if (layerName.Contains("BatchNorm"))
                    return "Accuracy sensitive";
                if (layerName.Contains("Embedding"))
                    return "Lookup operation";
                return "Default FP32";
            }

            if (precision == QuantizationType.Int16)
            {
                return "Early layer sensitivity";
            }

            return "Standard quantization";
        }

        /// <summary>
        /// Print the quantization configuration.
        /// </summary>
        private static void PrintConfiguration(QuantizationConfig config)
        {
            var stats = config.GetConfigurationStatistics();
            Console.WriteLine($"  Total layers: {stats.TotalLayers}");
            Console.WriteLine($"  Quantized layers: {stats.QuantizedLayers}");
            Console.WriteLine($"  FP32 layers: {stats.FP32Layers}");
            Console.WriteLine($"  Int8 layers: {stats.Int8Layers}");
            Console.WriteLine($"  Int16 layers: {stats.Int16Layers}\n");
        }

        /// <summary>
        /// Apply quantization with per-layer configuration.
        /// </summary>
        private static async Task<IModel> ApplyPerLayerQuantization(
            IModel model,
            IDataLoader calibrationData,
            QuantizationConfig config)
        {
            var quantizer = new PTQQuantizer();
            var quantizedModel = await quantizer.QuantizeAsync(model, calibrationData, config);
            return quantizedModel;
        }

        /// <summary>
        /// Analyze quantization distribution in the model.
        /// </summary>
        private static void AnalyzeQuantizationDistribution(IModel model)
        {
            Console.WriteLine("  Quantization Distribution:");

            var layers = model.GetAllLayers();
            var int8Count = 0;
            var int16Count = 0;
            var fp32Count = 0;

            foreach (var layer in layers)
            {
                var precision = layer.GetQuantizationPrecision();
                switch (precision)
                {
                    case QuantizationType.Int8:
                        int8Count++;
                        break;
                    case QuantizationType.Int16:
                        int16Count++;
                        break;
                    case QuantizationType.UInt8:
                        // Not used in this example
                        break;
                    default:
                        fp32Count++;
                        break;
                }
            }

            var total = layers.Count;
            Console.WriteLine($"    Int8:   {int8Count,3} layers ({int8Count * 100.0 / total:F1}%)");
            Console.WriteLine($"    Int16:  {int16Count,3} layers ({int16Count * 100.0 / total:F1}%)");
            Console.WriteLine($"    FP32:   {fp32Count,3} layers ({fp32Count * 100.0 / total:F1}%)");

            // Calculate memory savings
            var fp32Memory = total * 4; // Assume 4 bytes per parameter
            var quantizedMemory = int8Count * 1 + int16Count * 2 + fp32Count * 4;
            var savings = (1 - quantizedMemory / (double)fp32Memory) * 100;
            Console.WriteLine($"    Memory savings: {savings:F1}%");
        }

        /// <summary>
        /// Load model for quantization.
        /// </summary>
        private static async Task<IModel> LoadModel()
        {
            // In production, load a real pre-trained model
            var model = CreateSampleModel();
            return model;
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
        /// Prepare test data for evaluation.
        /// </summary>
        private static async Task<IDataLoader> PrepareTestData()
        {
            var testData = CreateSampleTestData();
            return testData;
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

        #region Sample Implementation (Replace with real code in production)

        private static IModel CreateSampleModel()
        {
            // Placeholder: Create a sample model
            throw new NotImplementedException("Replace with actual model creation code");
        }

        private static IDataLoader CreateSampleCalibrationData(int sampleCount)
        {
            // Placeholder: Create sample calibration data
            throw new NotImplementedException("Replace with actual data loading code");
        }

        private static IDataLoader CreateSampleTestData()
        {
            // Placeholder: Create sample test data
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

    #region Supporting Data Structures

    /// <summary>
    /// Layer-specific quantization configuration.
    /// </summary>
    internal class LayerQuantizationConfig
    {
        public bool Enabled { get; set; }
        public QuantizationType QuantizationType { get; set; }
        public QuantizationMode QuantizationMode { get; set; }
    }

    /// <summary>
    /// Configuration statistics.
    /// </summary>
    internal class ConfigurationStatistics
    {
        public int TotalLayers { get; set; }
        public int QuantizedLayers { get; set; }
        public int FP32Layers { get; set; }
        public int Int8Layers { get; set; }
        public int Int16Layers { get; set; }
    }

    /// <summary>
    /// Extensions for QuantizationConfig.
    /// </summary>
    internal static class QuantizationConfigExtensions
    {
        private static readonly Dictionary<string, LayerQuantizationConfig> LayerConfigs = new();

        public static void SetLayerConfiguration(
            this QuantizationConfig config,
            string layerName,
            LayerQuantizationConfig layerConfig)
        {
            LayerConfigs[layerName] = layerConfig;
        }

        public static LayerQuantizationConfig GetLayerConfiguration(
            this QuantizationConfig config,
            string layerName)
        {
            return LayerConfigs.TryGetValue(layerName, out var cfg) ? cfg : new LayerQuantizationConfig();
        }

        public static ConfigurationStatistics GetConfigurationStatistics(this QuantizationConfig config)
        {
            // This would be implemented with actual layer tracking
            return new ConfigurationStatistics
            {
                TotalLayers = LayerConfigs.Count,
                QuantizedLayers = LayerConfigs.Values.Count(c => c.Enabled),
                FP32Layers = LayerConfigs.Values.Count(c => !c.Enabled),
                Int8Layers = LayerConfigs.Values.Count(c => c.Enabled && c.QuantizationType == QuantizationType.Int8),
                Int16Layers = LayerConfigs.Values.Count(c => c.Enabled && c.QuantizationType == QuantizationType.Int16)
            };
        }
    }

    #endregion
}
