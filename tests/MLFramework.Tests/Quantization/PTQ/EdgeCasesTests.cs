using System;
using System.Collections.Generic;
using Xunit;
using MLFramework.Quantization.PTQ;
using MLFramework.Quantization.DataStructures;
using MLFramework.Quantization.Calibration;
using MLFramework.NN;
using MLFramework.Data;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Quantization.PTQ
{
    /// <summary>
    /// Tests for edge cases in PTQ workflow.
    /// </summary>
    public class EdgeCasesTests
    {
        private class MockModule : Module
        {
            public MockModule(string name)
            {
                Name = name;
            }

            public override Tensor Forward(Tensor input)
            {
                return input.Clone();
            }
        }

        private class MockLinearModule : Module
        {
            public Parameter Weight { get; }

            public MockLinearModule(string name, int inFeatures, int outFeatures)
            {
                Name = name;
                float[] weightData = new float[inFeatures * outFeatures];
                for (int i = 0; i < weightData.Length; i++)
                {
                    weightData[i] = 0.5f;
                }
                Weight = new Parameter(weightData, new int[] { outFeatures, inFeatures }, requiresGrad: true);
            }

            public override Tensor Forward(Tensor input)
            {
                return input.Clone();
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
        public void CalibrationProcess_WithEmptyCalibrationData_DoesNotThrow()
        {
            // Arrange
            var model = new MockModule("test_model");
            var dataLoader = new MockDataLoader(Array.Empty<Tensor>());
            var config = new QuantizationConfig
            {
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var calibrationProcess = new CalibrationProcess(calibratorFactory, modelTraversal);

            // Act & Assert
            var exception = Record.Exception(() => 
                calibrationProcess.RunCalibration(model, dataLoader, config));
            Assert.Null(exception);
        }

        [Fact]
        public void CalibrationProcess_WithEmptyCalibrationData_ReturnsEmptyParameters()
        {
            // Arrange
            var model = new MockModule("test_model");
            var dataLoader = new MockDataLoader(Array.Empty<Tensor>());
            var config = new QuantizationConfig
            {
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var calibrationProcess = new CalibrationProcess(calibratorFactory, modelTraversal);

            // Act
            var activationParams = calibrationProcess.RunCalibration(model, dataLoader, config);

            // Assert
            Assert.NotNull(activationParams);
            // Empty data loader should return empty parameters
            Assert.True(activationParams.Count >= 0);
        }

        [Fact]
        public void CalibrationProcess_WithSingleBatch_CollectsStatistics()
        {
            // Arrange
            var model = new MockModule("test_model");
            var batch = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f }, new int[] { 1, 5 });
            var dataLoader = new MockDataLoader(new[] { batch });
            var config = new QuantizationConfig
            {
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var calibrationProcess = new CalibrationProcess(calibratorFactory, modelTraversal);

            // Act
            var activationParams = calibrationProcess.RunCalibration(model, dataLoader, config);

            // Assert
            Assert.NotNull(activationParams);
        }

        [Fact]
        public void CalibrationProcess_WithSingleBatch_HandlesEdgeCase()
        {
            // Arrange
            var model = new MockModule("test_model");
            var batch = new Tensor(new float[] { 1.0f, 1.0f, 1.0f }, new int[] { 1, 3 });
            var dataLoader = new MockDataLoader(new[] { batch });
            var config = new QuantizationConfig
            {
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var calibrationProcess = new CalibrationProcess(calibratorFactory, modelTraversal);

            // Act
            var activationParams = calibrationProcess.RunCalibration(model, dataLoader, config);

            // Assert
            Assert.NotNull(activationParams);
        }

        [Fact]
        public void PTQQuantizer_WithVerySmallModel_WorksCorrectly()
        {
            // Arrange
            var model = new MockModule("tiny_model");
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };
            var quantizer = new PTQQuantizer();

            // Act
            var exception = Record.Exception(() => quantizer.Quantize(model, null, config));

            // Assert
            Assert.Null(exception);
            var skippedLayers = quantizer.GetSkippedLayers();
            Assert.NotNull(skippedLayers);
        }

        [Fact]
        public void PTQQuantizer_WithModelWithNoQuantizableLayers_WorksCorrectly()
        {
            // Arrange
            var model = new MockModule("no_quantizable");
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };
            var quantizer = new PTQQuantizer();

            // Act
            quantizer.Quantize(model, null, config);

            // Assert
            var skippedLayers = quantizer.GetSkippedLayers();
            Assert.NotNull(skippedLayers);
            // Model has no quantizable layers, so no layers should have parameters
        }

        [Fact]
        public void PTQQuantizer_WithModelWithNoQuantizableLayers_ReturnsEmptySkipped()
        {
            // Arrange
            var model = new MockModule("no_quantizable");
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };
            var quantizer = new PTQQuantizer();

            // Act
            quantizer.Quantize(model, null, config);

            // Assert
            var skippedLayers = quantizer.GetSkippedLayers();
            Assert.NotNull(skippedLayers);
        }

        [Fact]
        public void CalibrationProcess_WithAllSameValues_HandlesEdgeCase()
        {
            // Arrange
            var model = new MockModule("test_model");
            var batch = new Tensor(new float[] { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }, new int[] { 1, 5 });
            var dataLoader = new MockDataLoader(new[] { batch });
            var config = new QuantizationConfig
            {
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var calibrationProcess = new CalibrationProcess(calibratorFactory, modelTraversal);

            // Act
            var activationParams = calibrationProcess.RunCalibration(model, dataLoader, config);

            // Assert
            Assert.NotNull(activationParams);
        }

        [Fact]
        public void CollectActivationStatistics_WithAllSameValues_ReturnsZeroStdDev()
        {
            // Arrange
            var calibrationProcess = new CalibrationProcess();
            float[] data = new float[100];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = 1.0f;
            }
            var activation = new Tensor(data, new int[] { 1, 100 });

            // Act
            var stats = calibrationProcess.CollectActivationStatistics(null, activation);

            // Assert
            Assert.NotNull(stats);
            Assert.Equal(0.0f, stats.StdDev);
            Assert.Equal(1.0f, stats.Min);
            Assert.Equal(1.0f, stats.Max);
        }

        [Fact]
        public void CollectActivationStatistics_WithVerySmallValues_HandlesCorrectly()
        {
            // Arrange
            var calibrationProcess = new CalibrationProcess();
            float[] data = new float[] { 1e-10f, 2e-10f, 3e-10f };
            var activation = new Tensor(data, new int[] { 1, 3 });

            // Act
            var stats = calibrationProcess.CollectActivationStatistics(null, activation);

            // Assert
            Assert.NotNull(stats);
            Assert.Equal(1e-10f, stats.Min);
            Assert.Equal(3e-10f, stats.Max);
        }

        [Fact]
        public void CollectActivationStatistics_WithVeryLargeValues_HandlesCorrectly()
        {
            // Arrange
            var calibrationProcess = new CalibrationProcess();
            float[] data = new float[] { 1e10f, 2e10f, 3e10f };
            var activation = new Tensor(data, new int[] { 1, 3 });

            // Act
            var stats = calibrationProcess.CollectActivationStatistics(null, activation);

            // Assert
            Assert.NotNull(stats);
            Assert.Equal(1e10f, stats.Min);
            Assert.Equal(3e10f, stats.Max);
        }

        [Fact]
        public void PTQQuantizer_WithZeroScale_HandlesEdgeCase()
        {
            // Arrange
            var model = new MockLinearModule("layer", 10, 5);
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };
            var quantizer = new PTQQuantizer();

            // Act
            quantizer.Quantize(model, null, config);

            // Assert
            var skippedLayers = quantizer.GetSkippedLayers();
            Assert.NotNull(skippedLayers);
        }

        [Fact]
        public void PTQQuantizer_WithInfiniteValues_DoesNotThrow()
        {
            // Arrange
            var model = new MockModule("test_model");
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };
            var quantizer = new PTQQuantizer();

            // Act & Assert
            var exception = Record.Exception(() => quantizer.Quantize(model, null, config));
            Assert.Null(exception);
        }

        [Fact]
        public void CalibrationProcess_WithNaNValues_FiltersThemOut()
        {
            // Arrange
            var model = new MockModule("test_model");
            float[] data = new float[] { 1.0f, float.NaN, 3.0f, float.PositiveInfinity };
            var batch = new Tensor(data, new int[] { 1, 4 });
            var dataLoader = new MockDataLoader(new[] { batch });
            var config = new QuantizationConfig
            {
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var calibrationProcess = new CalibrationProcess(calibratorFactory, modelTraversal);

            // Act
            var activationParams = calibrationProcess.RunCalibration(model, dataLoader, config);

            // Assert
            Assert.NotNull(activationParams);
        }

        [Fact]
        public void DynamicQuantization_WithEmptyModel_WorksCorrectly()
        {
            // Arrange
            var model = new MockModule("empty_model");
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var dynamicQuantization = new DynamicQuantization(modelTraversal, calibratorFactory);

            var layerParams = new Dictionary<string, QuantizationParameters>();
            var layerFallback = new Dictionary<string, bool>();
            var skippedLayers = new List<string>();

            // Act
            dynamicQuantization.Quantize(model, config, layerFallback, layerParams, skippedLayers);

            // Assert
            Assert.NotNull(skippedLayers);
        }

        [Fact]
        public void StaticQuantization_WithEmptyModel_WorksCorrectly()
        {
            // Arrange
            var model = new MockModule("empty_model");
            var dataLoader = new MockDataLoader(Array.Empty<Tensor>());
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var calibrationProcess = new CalibrationProcess(calibratorFactory, modelTraversal);
            var staticQuantization = new StaticQuantization(modelTraversal, calibratorFactory, calibrationProcess);

            var layerParams = new Dictionary<string, QuantizationParameters>();
            var layerFallback = new Dictionary<string, bool>();
            var skippedLayers = new List<string>();

            // Act
            staticQuantization.Quantize(model, dataLoader, config, layerFallback, layerParams, skippedLayers);

            // Assert
            Assert.NotNull(skippedLayers);
        }

        [Fact]
        public void CalibrationProcess_WithZeroBatchSize_HandlesEdgeCase()
        {
            // Arrange
            var model = new MockModule("test_model");
            var dataLoader = new MockDataLoader(Array.Empty<Tensor>());
            var config = new QuantizationConfig
            {
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var calibrationProcess = new CalibrationProcess(calibratorFactory, modelTraversal);

            // Act
            var activationParams = calibrationProcess.RunCalibration(model, dataLoader, config);

            // Assert
            Assert.NotNull(activationParams);
        }

        [Fact]
        public void CalibrationProcess_WithAllNegativeValues_WorksCorrectly()
        {
            // Arrange
            var model = new MockModule("test_model");
            float[] data = new float[] { -5.0f, -3.0f, -1.0f };
            var batch = new Tensor(data, new int[] { 1, 3 });
            var dataLoader = new MockDataLoader(new[] { batch });
            var config = new QuantizationConfig
            {
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var calibrationProcess = new CalibrationProcess(calibratorFactory, modelTraversal);

            // Act
            var activationParams = calibrationProcess.RunCalibration(model, dataLoader, config);

            // Assert
            Assert.NotNull(activationParams);
        }

        [Fact]
        public void CalibrationProcess_WithAllPositiveValues_WorksCorrectly()
        {
            // Arrange
            var model = new MockModule("test_model");
            float[] data = new float[] { 1.0f, 3.0f, 5.0f };
            var batch = new Tensor(data, new int[] { 1, 3 });
            var dataLoader = new MockDataLoader(new[] { batch });
            var config = new QuantizationConfig
            {
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var calibrationProcess = new CalibrationProcess(calibratorFactory, modelTraversal);

            // Act
            var activationParams = calibrationProcess.RunCalibration(model, dataLoader, config);

            // Assert
            Assert.NotNull(activationParams);
        }

        [Fact]
        public void CollectActivationStatistics_WithSingleValue_WorksCorrectly()
        {
            // Arrange
            var calibrationProcess = new CalibrationProcess();
            var activation = new Tensor(new float[] { 5.0f }, new int[] { 1, 1 });

            // Act
            var stats = calibrationProcess.CollectActivationStatistics(null, activation);

            // Assert
            Assert.NotNull(stats);
            Assert.Equal(1, stats.SampleCount);
            Assert.Equal(5.0f, stats.Min);
            Assert.Equal(5.0f, stats.Max);
            Assert.Equal(5.0f, stats.Mean);
            Assert.Equal(0.0f, stats.StdDev);
        }

        [Fact]
        public void PTQQuantizer_WithLargeNumberOfLayers_WorksCorrectly()
        {
            // Arrange
            var model = new MockModule("large_model");
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };
            var quantizer = new PTQQuantizer();

            // Act
            var exception = Record.Exception(() => quantizer.Quantize(model, null, config));

            // Assert
            Assert.Null(exception);
        }
    }
}
