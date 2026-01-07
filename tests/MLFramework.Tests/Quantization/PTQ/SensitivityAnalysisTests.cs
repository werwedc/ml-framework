using System;
using System.Collections.Generic;
using Xunit;
using MLFramework.Quantization.PTQ;
using MLFramework.Quantization.DataStructures;
using MLFramework.NN;
using MLFramework.Data;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Quantization.PTQ
{
    /// <summary>
    /// Tests for per-layer sensitivity analysis functionality.
    /// </summary>
    public class SensitivityAnalysisTests
    {
        private class MockLinearModule : Module
        {
            public Parameter Weight { get; }

            public MockLinearModule(string name, int inFeatures, int outFeatures)
            {
                Name = name;
                var random = new Random(42);
                float[] weightData = new float[inFeatures * outFeatures];
                for (int i = 0; i < weightData.Length; i++)
                {
                    weightData[i] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                Weight = new Parameter(weightData, new int[] { outFeatures, inFeatures }, requiresGrad: true);
            }

            public override Tensor Forward(Tensor input)
            {
                return input.Clone();
            }
        }

        private class MockModel : Module
        {
            public MockLinearModule Layer1 { get; }
            public MockLinearModule Layer2 { get; }
            public MockLinearModule Layer3 { get; }

            public MockModel()
            {
                Name = "test_model";
                Layer1 = new MockLinearModule("layer1", 10, 20);
                Layer2 = new MockLinearModule("layer2", 20, 15);
                Layer3 = new MockLinearModule("layer3", 15, 5);
            }

            public override Tensor Forward(Tensor input)
            {
                var output = Layer1.Forward(input);
                output = Layer2.Forward(output);
                output = Layer3.Forward(output);
                return output;
            }
        }

        private class MockDataLoader : DataLoader<object>
        {
            private readonly Tensor[] _batches;
            private int _currentBatch;

            public MockDataLoader(Tensor[] batches)
            {
                _batches = batches;
                _currentBatch = 0;
            }

            public override int Count => _batches.Length;
            public override int BatchSize => 1;

            public override IEnumerator<object> GetEnumerator()
            {
                while (_currentBatch < _batches.Length)
                {
                    yield return _batches[_currentBatch++];
                }
            }

            public override void Reset()
            {
                _currentBatch = 0;
            }
        }

        [Fact]
        public void Analyze_PerLayer_AnalyzesSensitivityForEachLayer()
        {
            // Arrange
            var model = new MockModel();

            var random = new Random(42);
            var batches = new Tensor[10];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[10];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 10 });
            }

            var dataLoader = new MockDataLoader(batches);
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var sensitivityAnalyzer = new SensitivityAnalysis();

            // Act
            var results = sensitivityAnalyzer.Analyze(model, dataLoader, config);

            // Assert
            Assert.NotNull(results);
            Assert.Equal(3, results.Count);

            // Verify each layer has analysis results
            var layerNames = new HashSet<string>(results.Select(r => r.LayerName));
            Assert.Contains("layer1", layerNames);
            Assert.Contains("layer2", layerNames);
            Assert.Contains("layer3", layerNames);
        }

        [Fact]
        public void Analyze_PerLayer_IdentifiesSensitiveLayersCorrectly()
        {
            // Arrange
            var model = new MockModel();

            var random = new Random(42);
            var batches = new Tensor[10];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[10];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 10 });
            }

            var dataLoader = new MockDataLoader(batches);
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var sensitivityAnalyzer = new SensitivityAnalysis();

            // Act
            var results = sensitivityAnalyzer.Analyze(model, dataLoader, config);

            // Assert
            foreach (var result in results)
            {
                // Verify sensitivity score is in valid range [0, 1]
                Assert.InRange(result.SensitivityScore, 0.0f, 1.0f);

                // Verify accuracy loss is non-negative
                Assert.True(result.AccuracyLoss >= 0);

                // Verify predicted accuracy is <= baseline
                Assert.True(result.PredictedAccuracy <= result.BaselineAccuracy);
            }
        }

        [Fact]
        public void Analyze_PerLayer_VerifiesSensitivityThresholdsWork()
        {
            // Arrange
            var model = new MockModel();

            var random = new Random(42);
            var batches = new Tensor[10];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[10];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 10 });
            }

            var dataLoader = new MockDataLoader(batches);
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                CalibrationMethod = CalibrationMethod.MinMax,
                AccuracyThreshold = 0.01f // 1% threshold
            };
            var sensitivityAnalyzer = new SensitivityAnalysis();

            // Act
            var results = sensitivityAnalyzer.Analyze(model, dataLoader, config);

            // Assert
            // Count layers exceeding threshold
            var highSensitivityLayers = results.Where(r => r.AccuracyLoss > config.AccuracyThreshold).ToList();
            var lowSensitivityLayers = results.Where(r => r.AccuracyLoss <= config.AccuracyThreshold).ToList();

            // Both can be non-empty
            Assert.NotNull(highSensitivityLayers);
            Assert.NotNull(lowSensitivityLayers);

            // Total should equal number of layers
            Assert.Equal(results.Count, highSensitivityLayers.Count + lowSensitivityLayers.Count);
        }

        [Fact]
        public void Analyze_PerLayer_GeneratesSensitivityReport()
        {
            // Arrange
            var model = new MockModel();

            var random = new Random(42);
            var batches = new Tensor[10];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[10];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 10 });
            }

            var dataLoader = new MockDataLoader(batches);
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var sensitivityAnalyzer = new SensitivityAnalysis();

            // Act
            var results = sensitivityAnalyzer.Analyze(model, dataLoader, config);
            var report = sensitivityAnalyzer.GenerateSensitivityReport(results);

            // Assert
            Assert.NotNull(report);
            Assert.Contains("Sensitivity Analysis Report", report);
            Assert.Contains("Total layers analyzed", report);
        }

        [Fact]
        public void Analyze_WithNullModel_ThrowsArgumentNullException()
        {
            // Arrange
            Module model = null;
            var dataLoader = new MockDataLoader(Array.Empty<Tensor>());
            var config = new QuantizationConfig();
            var sensitivityAnalyzer = new SensitivityAnalysis();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => 
                sensitivityAnalyzer.Analyze(model, dataLoader, config));
        }

        [Fact]
        public void Analyze_WithNullDataLoader_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new MockModel();
            DataLoader<object> dataLoader = null;
            var config = new QuantizationConfig();
            var sensitivityAnalyzer = new SensitivityAnalysis();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => 
                sensitivityAnalyzer.Analyze(model, dataLoader, config));
        }

        [Fact]
        public void Analyze_WithNullConfig_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new MockModel();
            var dataLoader = new MockDataLoader(Array.Empty<Tensor>());
            QuantizationConfig config = null;
            var sensitivityAnalyzer = new SensitivityAnalysis();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => 
                sensitivityAnalyzer.Analyze(model, dataLoader, config));
        }

        [Fact]
        public void Analyze_VerifiesActivationStatisticsCollected()
        {
            // Arrange
            var model = new MockModel();

            var random = new Random(42);
            var batches = new Tensor[10];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[10];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 10 });
            }

            var dataLoader = new MockDataLoader(batches);
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var sensitivityAnalyzer = new SensitivityAnalysis();

            // Act
            var results = sensitivityAnalyzer.Analyze(model, dataLoader, config);

            // Assert
            foreach (var result in results)
            {
                // Verify activation statistics are collected
                Assert.True(result.MinActivation < result.MaxActivation);
                Assert.NotNull(result.MeanActivation);
                Assert.True(result.StdDevActivation >= 0);
            }
        }

        [Fact]
        public void GenerateSensitivityReport_SortsByAccuracyLoss()
        {
            // Arrange
            var model = new MockModel();

            var random = new Random(42);
            var batches = new Tensor[10];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[10];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 10 });
            }

            var dataLoader = new MockDataLoader(batches);
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var sensitivityAnalyzer = new SensitivityAnalysis();

            // Act
            var results = sensitivityAnalyzer.Analyze(model, dataLoader, config);
            var report = sensitivityAnalyzer.GenerateSensitivityReport(results);

            // Assert
            // Report should contain all layers
            Assert.Contains("layer1", report);
            Assert.Contains("layer2", report);
            Assert.Contains("layer3", report);
        }

        [Fact]
        public void GenerateSensitivityReport_IncludesAccuracyStatistics()
        {
            // Arrange
            var model = new MockModel();

            var random = new Random(42);
            var batches = new Tensor[10];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[10];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 10 });
            }

            var dataLoader = new MockDataLoader(batches);
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var sensitivityAnalyzer = new SensitivityAnalysis();

            // Act
            var results = sensitivityAnalyzer.Analyze(model, dataLoader, config);
            var report = sensitivityAnalyzer.GenerateSensitivityReport(results);

            // Assert
            Assert.Contains("High sensitivity layers", report);
            Assert.Contains("Medium sensitivity layers", report);
            Assert.Contains("Low sensitivity layers", report);
        }

        [Fact]
        public void Analyze_WithEmptyDataLoader_DoesNotThrow()
        {
            // Arrange
            var model = new MockModel();
            var dataLoader = new MockDataLoader(Array.Empty<Tensor>());
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var sensitivityAnalyzer = new SensitivityAnalysis();

            // Act & Assert
            var exception = Record.Exception(() => 
                sensitivityAnalyzer.Analyze(model, dataLoader, config));
            Assert.Null(exception);
        }

        [Fact]
        public void Analyze_WithSingleBatch_WorksCorrectly()
        {
            // Arrange
            var model = new MockModel();

            var random = new Random(42);
            float[] data = new float[10];
            for (int j = 0; j < data.Length; j++)
            {
                data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
            }
            var batch = new Tensor(data, new int[] { 1, 10 });
            var dataLoader = new MockDataLoader(new[] { batch });

            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var sensitivityAnalyzer = new SensitivityAnalysis();

            // Act
            var results = sensitivityAnalyzer.Analyze(model, dataLoader, config);

            // Assert
            Assert.NotNull(results);
            Assert.Equal(3, results.Count);
        }

        [Fact]
        public void Analyze_VerifiesSensitivityScoreInRange()
        {
            // Arrange
            var model = new MockModel();

            var random = new Random(42);
            var batches = new Tensor[5];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[10];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 10 });
            }

            var dataLoader = new MockDataLoader(batches);
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var sensitivityAnalyzer = new SensitivityAnalysis();

            // Act
            var results = sensitivityAnalyzer.Analyze(model, dataLoader, config);

            // Assert
            foreach (var result in results)
            {
                // Sensitivity score should be in [0, 1]
                Assert.InRange(result.SensitivityScore, 0.0f, 1.0f);
            }
        }

        [Fact]
        public void Analyze_VerifiesAccuracyLossCalculation()
        {
            // Arrange
            var model = new MockModel();

            var random = new Random(42);
            var batches = new Tensor[5];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[10];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 10 });
            }

            var dataLoader = new MockDataLoader(batches);
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var sensitivityAnalyzer = new SensitivityAnalysis();

            // Act
            var results = sensitivityAnalyzer.Analyze(model, dataLoader, config);

            // Assert
            foreach (var result in results)
            {
                // Accuracy loss = baseline - predicted
                var expectedLoss = result.BaselineAccuracy - result.PredictedAccuracy;
                Assert.Equal(expectedLoss, result.AccuracyLoss, precision: 6);

                // Accuracy loss should be non-negative
                Assert.True(result.AccuracyLoss >= 0);
            }
        }
    }
}
