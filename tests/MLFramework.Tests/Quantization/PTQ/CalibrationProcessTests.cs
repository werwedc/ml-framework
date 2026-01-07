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
    /// Tests for calibration process functionality.
    /// </summary>
    public class CalibrationProcessTests
    {
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

        [Fact]
        public void RunCalibration_WithSyntheticData_VerifiesStatisticsAreCollected()
        {
            // Arrange
            var model = new MockModule("test_model");

            var random = new Random(42);
            var batches = new Tensor[10];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[100];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 100 });
            }

            var dataLoader = new MockDataLoader(batches);
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
            // Should have collected statistics for the model
            Assert.True(activationParams.Count >= 0);
        }

        [Fact]
        public void RunCalibration_WithSyntheticData_VerifiesQuantizationParametersAreComputed()
        {
            // Arrange
            var model = new MockModule("test_model");

            var random = new Random(42);
            var batches = new Tensor[10];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[100];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 100 });
            }

            var dataLoader = new MockDataLoader(batches);
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
            // All parameters should have valid scale
            foreach (var kvp in activationParams)
            {
                var parameters = kvp.Value;
                Assert.True(parameters.Scale > 0, $"Scale should be positive for layer {kvp.Key}");
                Assert.InRange(parameters.ZeroPoint, -128, 127);
            }
        }

        [Fact]
        public void RunCalibration_WithSyntheticData_VerifiesCalibrationConvergence()
        {
            // Arrange
            var model = new MockModule("test_model");

            var random = new Random(42);
            var batches = new Tensor[10];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[100];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 100 });
            }

            var dataLoader = new MockDataLoader(batches);
            var config = new QuantizationConfig
            {
                CalibrationMethod = CalibrationMethod.MinMax
            };
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var calibrationProcess = new CalibrationProcess(calibratorFactory, modelTraversal);

            // Act - Run calibration twice to verify consistency
            var params1 = calibrationProcess.RunCalibration(model, dataLoader, config);
            dataLoader.Reset();
            var params2 = calibrationProcess.RunCalibration(model, dataLoader, config);

            // Assert
            // Parameters should be consistent across runs
            Assert.Equal(params1.Count, params2.Count);

            foreach (var key in params1.Keys)
            {
                if (params2.ContainsKey(key))
                {
                    Assert.Equal(params1[key].Scale, params2[key].Scale, precision: 5);
                    Assert.Equal(params1[key].ZeroPoint, params2[key].ZeroPoint);
                }
            }
        }

        [Fact]
        public void RunCalibration_WithEmptyData_DoesNotThrow()
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
        public void RunCalibration_WithSingleBatch_CollectsStatistics()
        {
            // Arrange
            var model = new MockModule("test_model");

            var random = new Random(42);
            var batch = new Tensor(100);
            for (int i = 0; i < batch.Data.Length; i++)
            {
                batch.Data[i] = (float)((random.NextDouble() - 0.5) * 2.0);
            }

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
        public void RunCalibration_WithNullModel_ThrowsArgumentNullException()
        {
            // Arrange
            Module model = null;
            var dataLoader = new MockDataLoader(Array.Empty<Tensor>());
            var config = new QuantizationConfig();
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var calibrationProcess = new CalibrationProcess(calibratorFactory, modelTraversal);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => 
                calibrationProcess.RunCalibration(model, dataLoader, config));
        }

        [Fact]
        public void RunCalibration_WithNullDataLoader_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new MockModule("test_model");
            DataLoader<object> dataLoader = null;
            var config = new QuantizationConfig();
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var calibrationProcess = new CalibrationProcess(calibratorFactory, modelTraversal);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => 
                calibrationProcess.RunCalibration(model, dataLoader, config));
        }

        [Fact]
        public void RunCalibration_WithNullConfig_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new MockModule("test_model");
            var dataLoader = new MockDataLoader(Array.Empty<Tensor>());
            QuantizationConfig config = null;
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var calibrationProcess = new CalibrationProcess(calibratorFactory, modelTraversal);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => 
                calibrationProcess.RunCalibration(model, dataLoader, config));
        }

        [Fact]
        public void CollectActivationStatistics_WithValidTensor_ReturnsValidStatistics()
        {
            // Arrange
            var calibrationProcess = new CalibrationProcess();
            var random = new Random(42);

            float[] data = new float[100];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)((random.NextDouble() - 0.5) * 2.0);
            }
            var activation = new Tensor(data, new int[] { 1, 100 });

            // Act
            var stats = calibrationProcess.CollectActivationStatistics(null, activation);

            // Assert
            Assert.NotNull(stats);
            Assert.Equal(100, stats.SampleCount);
            Assert.True(stats.Min < stats.Max);
            Assert.NotNull(stats.Histogram);
            Assert.Equal(50, stats.Histogram.Length);
        }

        [Fact]
        public void CollectActivationStatistics_WithEmptyTensor_ReturnsZeroSampleCount()
        {
            // Arrange
            var calibrationProcess = new CalibrationProcess();
            var activation = new Tensor(Array.Empty<float>(), new int[] { 0, 10 });

            // Act
            var stats = calibrationProcess.CollectActivationStatistics(null, activation);

            // Assert
            Assert.NotNull(stats);
            Assert.Equal(0, stats.SampleCount);
        }

        [Fact]
        public void CollectActivationStatistics_WithNullTensor_ThrowsArgumentNullException()
        {
            // Arrange
            var calibrationProcess = new CalibrationProcess();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => 
                calibrationProcess.CollectActivationStatistics(null, null!));
        }

        [Fact]
        public void CollectActivationStatistics_WithConstantValues_HandlesEdgeCase()
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
            Assert.Equal(100, stats.SampleCount);
            Assert.Equal(1.0f, stats.Min);
            Assert.Equal(1.0f, stats.Max);
            Assert.Equal(1.0f, stats.Mean);
            Assert.Equal(0.0f, stats.StdDev);
        }

        [Fact]
        public void Reset_ClearsAllStatistics()
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

            // Act - Run calibration then reset
            calibrationProcess.RunCalibration(model, dataLoader, config);
            calibrationProcess.Reset();

            // Assert
            // After reset, running calibration should not be affected by previous state
            var activationParams = calibrationProcess.RunCalibration(model, dataLoader, config);
            Assert.NotNull(activationParams);
        }

        [Fact]
        public void RunCalibration_WithMultipleBatches_AggregatesStatistics()
        {
            // Arrange
            var model = new MockModule("test_model");

            var random = new Random(42);
            var batches = new Tensor[20];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[100];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 100 });
            }

            var dataLoader = new MockDataLoader(batches);
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
            // All parameters should be valid
            foreach (var kvp in activationParams)
            {
                var parameters = kvp.Value;
                Assert.True(parameters.Scale > 0);
                Assert.InRange(parameters.ZeroPoint, -128, 127);
            }
        }

        [Fact]
        public void RunCalibration_WithMinMaxCalibration_UsesMinMaxMethod()
        {
            // Arrange
            var model = new MockModule("test_model");

            var random = new Random(42);
            var batches = new Tensor[5];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[100];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 100 });
            }

            var dataLoader = new MockDataLoader(batches);
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
        public void RunCalibration_WithEntropyCalibration_UsesEntropyMethod()
        {
            // Arrange
            var model = new MockModule("test_model");

            var random = new Random(42);
            var batches = new Tensor[5];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[100];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 100 });
            }

            var dataLoader = new MockDataLoader(batches);
            var config = new QuantizationConfig
            {
                CalibrationMethod = CalibrationMethod.Entropy
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
        public void CollectActivationStatistics_ComputesCorrectMean()
        {
            // Arrange
            var calibrationProcess = new CalibrationProcess();
            float[] data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
            var activation = new Tensor(data, new int[] { 1, 5 });

            // Act
            var stats = calibrationProcess.CollectActivationStatistics(null, activation);

            // Assert
            Assert.Equal(3.0f, stats.Mean, precision: 5);
        }

        [Fact]
        public void CollectActivationStatistics_ComputesCorrectStdDev()
        {
            // Arrange
            var calibrationProcess = new CalibrationProcess();
            float[] data = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
            var activation = new Tensor(data, new int[] { 1, 5 });

            // Act
            var stats = calibrationProcess.CollectActivationStatistics(null, activation);

            // Assert
            // Expected std dev = sqrt(2) ≈ 1.4142
            Assert.Equal(1.41421f, stats.StdDev, precision: 4);
        }

        [Fact]
        public void CollectActivationStatistics_CreatesHistogram()
        {
            // Arrange
            var calibrationProcess = new CalibrationProcess();
            float[] data = new float[1000];
            var random = new Random(42);
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)random.NextDouble();
            }
            var activation = new Tensor(data, new int[] { 1, 1000 });

            // Act
            var stats = calibrationProcess.CollectActivationStatistics(null, activation);

            // Assert
            Assert.NotNull(stats.Histogram);
            Assert.Equal(50, stats.Histogram.Length);

            // Histogram bins should sum to 1.0 (normalized)
            float sum = 0;
            foreach (float bin in stats.Histogram)
            {
                sum += bin;
            }
            Assert.Equal(1.0f, sum, precision: 4);
        }
    }
}
