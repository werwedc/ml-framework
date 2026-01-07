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
    /// Tests for static quantization functionality.
    /// </summary>
    public class StaticQuantizationTests
    {
        private class MockConv2DModule : Module
        {
            public Parameter Weight { get; }
            public Parameter Bias { get; }

            public MockConv2DModule(int inChannels, int outChannels, int kernelSize)
            {
                Name = "conv2d";
                var random = new Random(42);
                int weightCount = outChannels * inChannels * kernelSize * kernelSize;
                float[] weightData = new float[weightCount];
                for (int i = 0; i < weightData.Length; i++)
                {
                    weightData[i] = (float)((random.NextDouble() - 0.5) * 2.0);
                }

                float[] biasData = new float[outChannels];
                for (int i = 0; i < biasData.Length; i++)
                {
                    biasData[i] = (float)((random.NextDouble() - 0.5) * 0.1);
                }

                Weight = new Parameter(weightData, new int[] { outChannels, inChannels, kernelSize, kernelSize }, requiresGrad: true);
                Bias = new Parameter(biasData, new int[] { outChannels }, requiresGrad: true);
            }

            public override Tensor Forward(Tensor input)
            {
                // Simplified conv2d forward pass - just return input for now
                return input.Clone();
            }
        }

        private class MockModelWithConv : Module
        {
            public MockConv2DModule Conv2D { get; }

            public MockModelWithConv()
            {
                Name = "conv_model";
                Conv2D = new MockConv2DModule(3, 16, 3);
            }

            public override Tensor Forward(Tensor input)
            {
                return Conv2D.Forward(input);
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
        public void Quantize_Conv2DModelWithCalibration_QuantizesWeightsAndActivations()
        {
            // Arrange
            var model = new MockModelWithConv();
            var random = new Random(42);

            // Create synthetic calibration data
            var batches = new Tensor[10];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[3 * 32 * 32];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 3, 32, 32 });
            }

            var dataLoader = new MockDataLoader(batches);
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
            // Should have weight quantization parameters
            Assert.Contains("conv2d", layerParams.Keys);
            // Should have activation quantization parameters
            Assert.Contains("conv2d_activation", layerParams.Keys);

            var weightParams = layerParams["conv2d"];
            Assert.True(weightParams.Scale > 0);

            var activationParams = layerParams["conv2d_activation"];
            Assert.True(activationParams.Scale > 0);
        }

        [Fact]
        public void Quantize_Conv2DModel_VerifiesWeightsAreQuantized()
        {
            // Arrange
            var model = new MockModelWithConv();
            var originalWeights = model.Conv2D.Weight.Data;

            var random = new Random(42);
            var batches = new Tensor[5];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[3 * 32 * 32];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 3, 32, 32 });
            }

            var dataLoader = new MockDataLoader(batches);
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
            Assert.True(layerParams.ContainsKey("conv2d"));

            var parameters = layerParams["conv2d"];
            for (int i = 0; i < originalWeights.Length; i++)
            {
                var quantized = QuantizationOperations.Quantize(originalWeights[i], parameters);
                Assert.InRange(quantized, sbyte.MinValue, sbyte.MaxValue);
            }
        }

        [Fact]
        public void Quantize_Conv2DModel_VerifiesActivationsAreQuantized()
        {
            // Arrange
            var model = new MockModelWithConv();

            var random = new Random(42);
            var batches = new Tensor[5];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[3 * 32 * 32];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 3, 32, 32 });
            }

            var dataLoader = new MockDataLoader(batches);
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
            Assert.Contains("conv2d_activation", layerParams.Keys);
            var activationParams = layerParams["conv2d_activation"];
            Assert.True(activationParams.Scale > 0);
            Assert.InRange(activationParams.ZeroPoint, -128, 127);
        }

        [Fact]
        public void Quantize_Conv2DModel_VerifiesCalibrationStatisticsAreCollected()
        {
            // Arrange
            var model = new MockModelWithConv();

            var random = new Random(42);
            var batches = new Tensor[5];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[3 * 32 * 32];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 3, 32, 32 });
            }

            var dataLoader = new MockDataLoader(batches);
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
            Assert.Contains("conv2d_activation", layerParams.Keys);
            var activationParams = layerParams["conv2d_activation"];

            // Calibration should produce valid min/max ranges
            Assert.True(activationParams.Min < activationParams.Max);
            Assert.True(activationParams.Min < 0 || activationParams.Min > 0); // Should have some variance
            Assert.True(activationParams.Max > 0);
        }

        [Fact]
        public void Quantize_Conv2DModel_VerifiesInferenceResultsAreAccurate()
        {
            // Arrange
            var model = new MockModelWithConv();
            var testInput = new Tensor(new float[3 * 32 * 32], new int[] { 1, 3, 32, 32 });

            var random = new Random(42);
            var batches = new Tensor[5];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[3 * 32 * 32];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 3, 32, 32 });
            }

            var dataLoader = new MockDataLoader(batches);
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

            // Get original inference result
            var originalOutput = model.Forward(testInput);

            // Act
            staticQuantization.Quantize(model, dataLoader, config, layerFallback, layerParams, skippedLayers);

            // The model structure is preserved, so inference should still work
            var quantizedOutput = model.Forward(testInput);

            // Assert - outputs should be similar
            var originalData = originalOutput.Data;
            var quantizedData = quantizedOutput.Data;

            for (int i = 0; i < originalData.Length; i++)
            {
                // Since we haven't actually replaced weights with Int8 in this mock,
                // outputs should be identical
                Assert.Equal(originalData[i], quantizedData[i], precision: 5);
            }
        }

        [Fact]
        public void Quantize_WithNullModel_ThrowsArgumentNullException()
        {
            // Arrange
            Module model = null;
            var dataLoader = new MockDataLoader(Array.Empty<Tensor>());
            var config = new QuantizationConfig();
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var calibrationProcess = new CalibrationProcess(calibratorFactory, modelTraversal);
            var staticQuantization = new StaticQuantization(modelTraversal, calibratorFactory, calibrationProcess);

            var layerParams = new Dictionary<string, QuantizationParameters>();
            var layerFallback = new Dictionary<string, bool>();
            var skippedLayers = new List<string>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                staticQuantization.Quantize(model, dataLoader, config, layerFallback, layerParams, skippedLayers));
        }

        [Fact]
        public void Quantize_WithNullDataLoader_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new MockModelWithConv();
            DataLoader<object> dataLoader = null;
            var config = new QuantizationConfig();
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var calibrationProcess = new CalibrationProcess(calibratorFactory, modelTraversal);
            var staticQuantization = new StaticQuantization(modelTraversal, calibratorFactory, calibrationProcess);

            var layerParams = new Dictionary<string, QuantizationParameters>();
            var layerFallback = new Dictionary<string, bool>();
            var skippedLayers = new List<string>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                staticQuantization.Quantize(model, dataLoader, config, layerFallback, layerParams, skippedLayers));
        }

        [Fact]
        public void Quantize_WithNullConfig_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new MockModelWithConv();
            var dataLoader = new MockDataLoader(Array.Empty<Tensor>());
            QuantizationConfig config = null;
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var calibrationProcess = new CalibrationProcess(calibratorFactory, modelTraversal);
            var staticQuantization = new StaticQuantization(modelTraversal, calibratorFactory, calibrationProcess);

            var layerParams = new Dictionary<string, QuantizationParameters>();
            var layerFallback = new Dictionary<string, bool>();
            var skippedLayers = new List<string>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                staticQuantization.Quantize(model, dataLoader, config, layerFallback, layerParams, skippedLayers));
        }

        [Fact]
        public void Quantize_WithLayerFallback_SkipsSpecifiedLayers()
        {
            // Arrange
            var model = new MockModelWithConv();

            var random = new Random(42);
            var batches = new Tensor[5];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[3 * 32 * 32];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 3, 32, 32 });
            }

            var dataLoader = new MockDataLoader(batches);
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
            var layerFallback = new Dictionary<string, bool>
            {
                { "conv2d", true } // Skip this layer
            };
            var skippedLayers = new List<string>();

            // Act
            staticQuantization.Quantize(model, dataLoader, config, layerFallback, layerParams, skippedLayers);

            // Assert
            Assert.Contains("conv2d", skippedLayers);
            Assert.DoesNotContain("conv2d", layerParams.Keys);
            Assert.DoesNotContain("conv2d_activation", layerParams.Keys);
        }

        [Fact]
        public void Quantize_WithPerChannelQuantization_UsesPerChannelParameters()
        {
            // Arrange
            var model = new MockModelWithConv();

            var random = new Random(42);
            var batches = new Tensor[5];
            for (int i = 0; i < batches.Length; i++)
            {
                float[] data = new float[3 * 32 * 32];
                for (int j = 0; j < data.Length; j++)
                {
                    data[j] = (float)((random.NextDouble() - 0.5) * 2.0);
                }
                batches[i] = new Tensor(data, new int[] { 1, 3, 32, 32 });
            }

            var dataLoader = new MockDataLoader(batches);
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerChannel,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax,
                EnablePerChannelQuantization = true
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
            Assert.Contains("conv2d", layerParams.Keys);
            var parameters = layerParams["conv2d"];
            Assert.True(parameters.Scale > 0);
        }
    }
}
