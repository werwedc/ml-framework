using MLFramework.Models;
using MLFramework.Data;
using MLFramework.Quantization.PTQ;
using MLFramework.Quantization.DataStructures;
using MLFramework.Quantization.Backends;

namespace MLFramework.Examples.Quantization
{
    /// <summary>
    /// Backend Selection Example demonstrates how to work with different quantization backends.
    /// This example shows how to check available backends, select optimal backend, compare performance,
    /// and switch backends dynamically.
    /// </summary>
    public class BackendSelection
    {
        /// <summary>
        /// Run the backend selection example.
        /// </summary>
        public static async Task Run()
        {
            Console.WriteLine("=== Backend Selection Example ===\n");

            // Step 1: Check available backends
            Console.WriteLine("Step 1: Checking available quantization backends...");
            var availableBackends = await CheckAvailableBackends();
            PrintAvailableBackends(availableBackends);

            // Step 2: Load model and data
            Console.WriteLine("\nStep 2: Loading model and test data...");
            var model = await LoadModel();
            var testData = await PrepareTestData();
            Console.WriteLine($"  Model: {model.GetType().Name}");
            Console.WriteLine($"  Test samples: {testData.GetSampleCount():,}\n");

            // Step 3: Compare backend performance
            Console.WriteLine("Step 3: Comparing backend performance...");
            var performanceResults = await CompareBackendPerformance(model, testData);
            PrintPerformanceResults(performanceResults);

            // Step 4: Select optimal backend
            Console.WriteLine("\nStep 4: Selecting optimal backend...");
            var optimalBackend = SelectOptimalBackend(availableBackends, performanceResults);
            Console.WriteLine($"  Optimal backend: {optimalBackend.GetName()}");
            Console.WriteLine($"  Reason: {optimalBackend.SelectionReason}\n");

            // Step 5: Test dynamic backend switching
            Console.WriteLine("Step 5: Testing dynamic backend switching...");
            await TestDynamicSwitching(model, testData);

            // Step 6: Demonstrate backend capabilities
            Console.WriteLine("\nStep 6: Demonstrating backend capabilities...");
            await DemonstrateBackendCapabilities(model, testData);

            // Step 7: Save backend configuration
            Console.WriteLine("\nStep 7: Saving backend configuration...");
            await SaveBackendConfiguration(optimalBackend);
            Console.WriteLine("  Configuration saved to: backend_config.json");

            Console.WriteLine("\n=== Backend Selection Example Complete ===");
        }

        /// <summary>
        /// Check which backends are available on the system.
        /// </summary>
        private static async Task<List<BackendInfo>> CheckAvailableBackends()
        {
            var backends = new List<BackendInfo>();

            // Check CPU Backend (always available)
            var cpuBackend = new CPUBackend();
            backends.Add(new BackendInfo
            {
                Name = cpuBackend.GetName(),
                IsAvailable = cpuBackend.IsAvailable(),
                Description = "Pure C# implementation with SIMD optimization",
                Type = "CPU"
            });

            // Check x86 Backend (Intel oneDNN)
            try
            {
                var x86Backend = new x86Backend();
                backends.Add(new BackendInfo
                {
                    Name = x86Backend.GetName(),
                    IsAvailable = x86Backend.IsAvailable(),
                    Description = "Intel oneDNN with AVX-512 and VNNI support",
                    Type = "x86"
                });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  x86 Backend: Not available ({ex.Message})");
            }

            // Check ARM Backend (ARM NEON)
            try
            {
                var armBackend = new ARMBackend();
                backends.Add(new BackendInfo
                {
                    Name = armBackend.GetName(),
                    IsAvailable = armBackend.IsAvailable(),
                    Description = "ARM NEON with ARMv8.2 dot product instructions",
                    Type = "ARM"
                });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ARM Backend: Not available ({ex.Message})");
            }

            // Check GPU Backend (CUDA)
            try
            {
                var gpuBackend = new GPUBackend();
                backends.Add(new BackendInfo
                {
                    Name = gpuBackend.GetName(),
                    IsAvailable = gpuBackend.IsAvailable(),
                    Description = "CUDA Tensor Cores with Int8 acceleration",
                    Type = "GPU"
                });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  GPU Backend: Not available ({ex.Message})");
            }

            return backends;
        }

        /// <summary>
        /// Print available backends.
        /// </summary>
        private static void PrintAvailableBackends(List<BackendInfo> backends)
        {
            Console.WriteLine("  Available Backends:");
            Console.WriteLine($"  Backend           | Available | Description");
            Console.WriteLine(new string('-', 80));

            foreach (var backend in backends)
            {
                var available = backend.IsAvailable ? "Yes" : "No";
                Console.WriteLine($"  {backend.Name,-17} | {available,9} | {backend.Description}");
            }
        }

        /// <summary>
        /// Compare performance of all available backends.
        /// </summary>
        private static async Task<List<BackendPerformance>> CompareBackendPerformance(
            IModel model,
            IDataLoader testData)
        {
            var results = new List<BackendPerformance>();
            var availableBackends = BackendFactory.GetAvailableBackends();

            Console.WriteLine("  Running performance benchmarks...");
            foreach (var backendName in availableBackends)
            {
                Console.Write($"    {backendName}... ");

                try
                {
                    var backend = BackendFactory.Create(backendName);
                    var performance = await MeasureBackendPerformance(model, backend, testData);

                    results.Add(new BackendPerformance
                    {
                        BackendName = backendName,
                        InferenceTimeMs = performance.AverageTimeMs,
                        ThroughputSamplesPerSec = performance.Throughput,
                        Accuracy = performance.Accuracy,
                        MemoryUsageMB = performance.MemoryUsageMB
                    });

                    Console.WriteLine($"{performance.AverageTimeMs:F2}ms, {performance.Throughput:F0} samples/s");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error: {ex.Message}");
                }
            }

            return results;
        }

        /// <summary>
        /// Measure performance of a specific backend.
        /// </summary>
        private static async Task<PerformanceMetrics> MeasureBackendPerformance(
            IModel model,
            IQuantizationBackend backend,
            IDataLoader testData)
        {
            var times = new List<float>();
            var totalSamples = 0;
            var accuracySum = 0f;
            var batchCount = 0;

            // Run inference on test data
            await foreach (var batch in testData)
            {
                var stopwatch = System.Diagnostics.Stopwatch.StartNew();

                // Run inference with this backend
                var outputs = backend.Infer(model, batch.Inputs);

                stopwatch.Stop();
                times.Add(stopwatch.ElapsedMilliseconds);
                totalSamples += batch.Inputs.Shape[0];

                // Calculate accuracy (simplified)
                var accuracy = CalculateAccuracy(outputs, batch.Targets);
                accuracySum += accuracy;
                batchCount++;

                // Limit to 10 batches for benchmarking
                if (batchCount >= 10) break;
            }

            return new PerformanceMetrics
            {
                AverageTimeMs = times.Average(),
                Throughput = totalSamples / times.Sum() * 1000,
                Accuracy = accuracySum / batchCount,
                MemoryUsageMB = EstimateBackendMemoryUsage(backend)
            };
        }

        /// <summary>
        /// Calculate accuracy (simplified).
        /// </summary>
        private static float CalculateAccuracy(Tensor<float> outputs, Tensor<float> targets)
        {
            // Placeholder: Calculate actual accuracy
            return 0.95f;
        }

        /// <summary>
        /// Estimate backend memory usage.
        /// </summary>
        private static float EstimateBackendMemoryUsage(IQuantizationBackend backend)
        {
            // Placeholder: Estimate actual memory usage
            return 100.0f;
        }

        /// <summary>
        /// Print performance results.
        /// </summary>
        private static void PrintPerformanceResults(List<BackendPerformance> results)
        {
            Console.WriteLine("  Performance Results:");
            Console.WriteLine($"  Backend           | Inference (ms) | Throughput (samples/s) | Accuracy");
            Console.WriteLine(new string('-', 80));

            foreach (var result in results.OrderBy(r => r.InferenceTimeMs))
            {
                Console.WriteLine($"  {result.BackendName,-17} | {result.InferenceTimeMs,13:F2} | {result.ThroughputSamplesPerSec,21:F0} | {result.Accuracy:F4}");
            }
        }

        /// <summary>
        /// Select optimal backend based on performance and availability.
        /// </summary>
        private static SelectedBackend SelectOptimalBackend(
            List<BackendInfo> availableBackends,
            List<BackendPerformance> performanceResults)
        {
            // Find best performing backend
            var bestPerformer = performanceResults
                .OrderBy(r => r.InferenceTimeMs)
                .FirstOrDefault();

            if (bestPerformer != null)
            {
                return new SelectedBackend
                {
                    BackendName = bestPerformer.BackendName,
                    SelectionReason = $"Best inference time: {bestPerformer.InferenceTimeMs:F2}ms"
                };
            }

            // Fallback to CPU backend
            return new SelectedBackend
            {
                BackendName = "CPU",
                SelectionReason = "Default backend (always available)"
            };
        }

        /// <summary>
        /// Test dynamic backend switching.
        /// </summary>
        private static async Task TestDynamicSwitching(IModel model, IDataLoader testData)
        {
            Console.WriteLine("  Testing backend switching...");

            // Start with CPU backend
            var factory = BackendFactory.CreateDefault();
            Console.WriteLine($"    Initial backend: {factory.GetName()}");

            // Switch to another backend if available
            var availableBackends = BackendFactory.GetAvailableBackends();
            if (availableBackends.Count > 1)
            {
                var newBackendName = availableBackends.First(b => b != factory.GetName());
                factory = BackendFactory.Create(newBackendName);
                Console.WriteLine($"    Switched to: {factory.GetName()}");

                // Run inference with new backend
                await RunInferenceWithBackend(model, factory, testData);
            }
        }

        /// <summary>
        /// Run inference with a specific backend.
        /// </summary>
        private static async Task RunInferenceWithBackend(
            IModel model,
            IQuantizationBackend backend,
            IDataLoader testData)
        {
            Console.WriteLine($"    Running inference with {backend.GetName()}...");
            await foreach (var batch in testData)
            {
                var outputs = backend.Infer(model, batch.Inputs);
                // Process outputs...
                break; // Just one batch for testing
            }
            Console.WriteLine("    Inference successful!");
        }

        /// <summary>
        /// Demonstrate backend capabilities.
        /// </summary>
        private static async Task DemonstrateBackendCapabilities(IModel model, IDataLoader testData)
        {
            var availableBackends = BackendFactory.GetAvailableBackends();

            Console.WriteLine("  Backend Capabilities:");
            Console.WriteLine($"  Backend           | Int8 MatMul | Int8 Conv2D | Per-Channel | Mixed Precision");
            Console.WriteLine(new string('-', 80));

            foreach (var backendName in availableBackends)
            {
                try
                {
                    var backend = BackendFactory.Create(backendName);
                    var capabilities = backend.GetCapabilities();

                    var int8MatMul = (capabilities.Flags & BackendCapabilityFlags.Int8MatMul) != 0 ? "Yes" : "No";
                    var int8Conv2D = (capabilities.Flags & BackendCapabilityFlags.Int8Conv2D) != 0 ? "Yes" : "No";
                    var perChannel = (capabilities.Flags & BackendCapabilityFlags.PerChannelQuantization) != 0 ? "Yes" : "No";
                    var mixedPrec = (capabilities.Flags & BackendCapabilityFlags.MixedPrecision) != 0 ? "Yes" : "No";

                    Console.WriteLine($"  {backendName,-17} | {int8MatMul,11} | {int8Conv2D,10} | {perChannel,10} | {mixedPrec,13}");
                }
                catch (Exception)
                {
                    Console.WriteLine($"  {backendName,-17} | N/A         | N/A        | N/A        | N/A");
                }
            }
        }

        /// <summary>
        /// Save backend configuration.
        /// </summary>
        private static async Task SaveBackendConfiguration(SelectedBackend backend)
        {
            var config = new BackendConfiguration
            {
                PreferredBackend = backend.BackendName,
                SelectionReason = backend.SelectionReason,
                Timestamp = DateTime.UtcNow
            };

            var json = System.Text.Json.JsonSerializer.Serialize(config, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true
            });
            await File.WriteAllTextAsync("backend_config.json", json);
        }

        #region Data Loading

        private static async Task<IModel> LoadModel()
        {
            var model = CreateSampleModel();
            return model;
        }

        private static async Task<IDataLoader> PrepareTestData()
        {
            var testData = CreateSampleTestData();
            return testData;
        }

        #endregion

        #region Data Structures

        private class BackendInfo
        {
            public string Name { get; set; } = string.Empty;
            public bool IsAvailable { get; set; }
            public string Description { get; set; } = string.Empty;
            public string Type { get; set; } = string.Empty;
        }

        private class BackendPerformance
        {
            public string BackendName { get; set; } = string.Empty;
            public float InferenceTimeMs { get; set; }
            public float ThroughputSamplesPerSec { get; set; }
            public float Accuracy { get; set; }
            public float MemoryUsageMB { get; set; }
        }

        private class PerformanceMetrics
        {
            public float AverageTimeMs { get; set; }
            public float Throughput { get; set; }
            public float Accuracy { get; set; }
            public float MemoryUsageMB { get; set; }
        }

        private class SelectedBackend
        {
            public string BackendName { get; set; } = string.Empty;
            public string SelectionReason { get; set; } = string.Empty;
        }

        private class BackendConfiguration
        {
            public string PreferredBackend { get; set; } = string.Empty;
            public string SelectionReason { get; set; } = string.Empty;
            public DateTime Timestamp { get; set; }
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
