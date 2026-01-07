using MLFramework.Models;
using MLFramework.Data;
using MLFramework.Quantization.PTQ;
using MLFramework.Quantization.DataStructures;
using MLFramework.Quantization.Calibration;
using MLFramework.Quantization.Evaluation;

namespace MLFramework.Examples.Quantization
{
    /// <summary>
    /// Custom Calibration Example demonstrates how to implement and use custom calibration methods.
    /// This example shows the comparison between different calibration strategies and how to create
    /// a custom calibrator for specific use cases.
    /// </summary>
    public class CustomCalibration
    {
        /// <summary>
        /// Run the custom calibration comparison example.
        /// </summary>
        public static async Task Run()
        {
            Console.WriteLine("=== Custom Calibration Example ===\n");

            // Step 1: Load model and data
            Console.WriteLine("Step 1: Loading model and calibration data...");
            var model = await LoadModel();
            var calibrationData = await PrepareCalibrationData();
            var testData = await PrepareTestData();
            Console.WriteLine($"  Model: {model.GetType().Name}\n");

            // Step 2: Compare standard calibration methods
            Console.WriteLine("Step 2: Comparing standard calibration methods...");
            var results = await CompareCalibrationMethods(model, calibrationData, testData);

            // Step 3: Display comparison results
            Console.WriteLine("\nStep 3: Calibration Method Comparison:");
            Console.WriteLine($"  Method              | Accuracy | Time (ms)");
            Console.WriteLine(new string('-', 50));
            foreach (var result in results)
            {
                Console.WriteLine($"  {result.Method,-18} | {result.Accuracy:F4}   | {result.TimeMs:F0}");
            }

            // Step 4: Demonstrate custom calibrator
            Console.WriteLine("\nStep 4: Demonstrating custom calibrator...");
            await DemonstrateCustomCalibrator(model, calibrationData, testData);

            // Step 5: Analyze calibration statistics
            Console.WriteLine("\nStep 5: Analyzing calibration statistics...");
            await AnalyzeCalibrationStatistics(model, calibrationData);

            Console.WriteLine("\n=== Custom Calibration Example Complete ===");
        }

        /// <summary>
        /// Compare different standard calibration methods.
        /// </summary>
        private static async Task<List<CalibrationResult>> CompareCalibrationMethods(
            IModel model,
            IDataLoader calibrationData,
            IDataLoader testData)
        {
            var methods = new[]
            {
                CalibrationMethod.MinMax,
                CalibrationMethod.Entropy,
                CalibrationMethod.Percentile,
                CalibrationMethod.MovingAverage
            };

            var results = new List<CalibrationResult>();

            foreach (var method in methods)
            {
                var stopwatch = System.Diagnostics.Stopwatch.StartNew();

                // Configure quantization with this calibration method
                var config = new QuantizationConfig
                {
                    WeightQuantization = QuantizationMode.PerTensorSymmetric,
                    ActivationQuantization = QuantizationMode.PerTensorAsymmetric,
                    CalibrationMethod = method,
                    CalibrationBatchSize = 32,
                    QuantizationType = QuantizationType.Int8,
                    FallbackToFP32 = true
                };

                // Apply PTQ
                var quantizer = new PTQQuantizer();
                var quantizedModel = await quantizer.QuantizeAsync(model, calibrationData, config);

                // Evaluate
                var accuracy = await EvaluateModel(quantizedModel, testData);

                stopwatch.Stop();

                results.Add(new CalibrationResult
                {
                    Method = method.ToString(),
                    Accuracy = accuracy,
                    TimeMs = stopwatch.ElapsedMilliseconds
                });

                Console.WriteLine($"  {method}: {accuracy:F4} ({stopwatch.ElapsedMilliseconds}ms)");
            }

            return results;
        }

        /// <summary>
        /// Demonstrate a custom calibrator implementation.
        /// </summary>
        private static async Task DemonstrateCustomCalibrator(
            IModel model,
            IDataLoader calibrationData,
            IDataLoader testData)
        {
            // Create custom calibrator (e.g., percentile with outlier removal)
            var customCalibrator = new RobustPercentileCalibrator(
                percentile: 99.5f,
                outlierRemoval: true
            );

            // Configure quantization with custom calibrator
            var config = new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerTensorSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorAsymmetric,
                CalibrationMethod = CalibrationMethod.Percentile, // Will use custom calibrator
                CalibrationBatchSize = 32,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = true
            };

            // Apply PTQ with custom calibrator
            var quantizer = new PTQQuantizer(customCalibrator);
            var quantizedModel = await quantizer.QuantizeAsync(model, calibrationData, config);

            // Evaluate
            var accuracy = await EvaluateModel(quantizedModel, testData);

            Console.WriteLine($"  Custom calibrator accuracy: {accuracy:F4}");
            Console.WriteLine($"  Percentile: 99.5%");
            Console.WriteLine($"  Outlier removal: Enabled");
        }

        /// <summary>
        /// Analyze calibration statistics for each layer.
        /// </summary>
        private static async Task AnalyzeCalibrationStatistics(
            IModel model,
            IDataLoader calibrationData)
        {
            var minMaxCalibrator = new MinMaxCalibrator();
            var entropyCalibrator = new EntropyCalibrator();

            Console.WriteLine("  Calibration Statistics by Layer:");
            Console.WriteLine($"  Layer                    | Min    | Max    | MinMax | Entropy");
            Console.WriteLine(new string('-', 70));

            // Collect calibration statistics for each layer
            var layerNames = model.GetQuantizableLayerNames();

            foreach (var layerName in layerNames.Take(5)) // Show first 5 layers as example
            {
                // Get layer activations from calibration data
                var activations = await GetLayerActivations(model, calibrationData, layerName);

                // Calibrate with different methods
                minMaxCalibrator.CollectStatistics(activations);
                var minMaxParams = minMaxCalibrator.GetQuantizationParameters();
                minMaxCalibrator.Reset();

                entropyCalibrator.CollectStatistics(activations);
                var entropyParams = entropyCalibrator.GetQuantizationParameters();
                entropyCalibrator.Reset();

                var minVal = activations.Min();
                var maxVal = activations.Max();

                Console.WriteLine($"  {layerName,-24} | {minVal,6:F2} | {maxVal,6:F2} | {minMaxParams.Scale,6:F4} | {entropyParams.Scale,7:F4}");
            }

            if (layerNames.Count > 5)
            {
                Console.WriteLine($"  ... and {layerNames.Count - 5} more layers");
            }
        }

        /// <summary>
        /// Load model for calibration.
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
            // Use a subset of training data (typically 100-500 samples)
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

        /// <summary>
        /// Get activations for a specific layer during calibration.
        /// </summary>
        private static async Task<float[]> GetLayerActivations(
            IModel model,
            IDataLoader calibrationData,
            string layerName)
        {
            var activations = new List<float>();

            // Run inference on calibration data and collect layer outputs
            // In production, this would use a forward hook
            await foreach (var batch in calibrationData)
            {
                var layerOutput = model.GetLayerOutput(batch.Inputs, layerName);
                activations.AddRange(layerOutput.Flatten().ToArray());
            }

            return activations.ToArray();
        }

        #region Custom Calibrator Implementation

        /// <summary>
        /// Custom calibrator that uses percentile with outlier removal.
        /// </summary>
        private class RobustPercentileCalibrator : ICalibrator
        {
            private readonly float _percentile;
            private readonly bool _outlierRemoval;
            private readonly List<float> _data;

            public RobustPercentileCalibrator(float percentile = 99.5f, bool outlierRemoval = true)
            {
                _percentile = percentile;
                _outlierRemoval = outlierRemoval;
                _data = new List<float>();
            }

            public void CollectStatistics(float[] data)
            {
                _data.AddRange(data);

                if (_outlierRemoval)
                {
                    RemoveOutliers();
                }
            }

            public QuantizationParameters GetQuantizationParameters()
            {
                if (_data.Count == 0)
                {
                    return new QuantizationParameters { Scale = 1.0f, ZeroPoint = 0 };
                }

                // Use percentile instead of min-max
                var sortedData = _data.OrderBy(x => x).ToArray();
                var minIndex = (int)((1 - _percentile / 100.0f) * sortedData.Length);
                var maxIndex = (int)(_percentile / 100.0f * sortedData.Length) - 1;

                var minVal = sortedData[Math.Max(0, minIndex)];
                var maxVal = sortedData[Math.Min(sortedData.Length - 1, maxIndex)];

                // Calculate scale and zero-point
                var qmin = -128f; // Int8 min
                var qmax = 127f;  // Int8 max

                var scale = (maxVal - minVal) / (qmax - qmin);
                var zeroPoint = (int)Math.Round(qmin - minVal / scale);

                return new QuantizationParameters
                {
                    Scale = scale,
                    ZeroPoint = zeroPoint,
                    Min = minVal,
                    Max = maxVal
                };
            }

            public void Reset()
            {
                _data.Clear();
            }

            private void RemoveOutliers()
            {
                if (_data.Count < 10) return;

                // Remove extreme outliers beyond 3 standard deviations
                var mean = _data.Average();
                var std = Math.Sqrt(_data.Average(x => Math.Pow(x - mean, 2)));

                var threshold = 3 * std;
                _data.RemoveAll(x => Math.Abs(x - mean) > threshold);
            }
        }

        #endregion

        #region Data Structures

        private class CalibrationResult
        {
            public string Method { get; set; } = string.Empty;
            public float Accuracy { get; set; }
            public long TimeMs { get; set; }
        }

        #endregion

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
}
