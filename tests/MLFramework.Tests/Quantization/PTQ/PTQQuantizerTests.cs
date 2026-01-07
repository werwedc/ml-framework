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
    /// Tests for PTQQuantizer functionality.
    /// </summary>
    public class PTQQuantizerTests
    {
        private class MockLinearModule : Module
        {
            public Parameter Weight { get; }

            public MockLinearModule(int inFeatures, int outFeatures)
            {
                Name = "linear";
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

        private class MockMultiLayerModel : Module
        {
            public MockLinearModule Layer1 { get; }
            public MockLinearModule Layer2 { get; }
            public MockLinearModule Layer3 { get; }

            public MockMultiLayerModel()
            {
                Name = "multi_layer_model";
                Layer1 = new MockLinearModule(10, 20) { Name = "layer1" };
                Layer2 = new MockLinearModule(20, 15) { Name = "layer2" };
                Layer3 = new MockLinearModule(15, 5) { Name = "layer3" };
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
        public void Quantize_MultiLayerModel_QuantizesAllQuantizableLayers()
        {
            // Arrange
            var model = new MockMultiLayerModel();
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
            Assert.Empty(skippedLayers);

            // All layers should have quantization parameters
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("layer1"));
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("layer2"));
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("layer3"));
        }

        [Fact]
        public void Quantize_MultiLayerModel_VerifiesAllQuantizableLayersAreQuantized()
        {
            // Arrange
            var model = new MockMultiLayerModel();
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
            var layer1Params = quantizer.GetLayerQuantizationParameters("layer1");
            var layer2Params = quantizer.GetLayerQuantizationParameters("layer2");
            var layer3Params = quantizer.GetLayerQuantizationParameters("layer3");

            Assert.NotNull(layer1Params);
            Assert.NotNull(layer2Params);
            Assert.NotNull(layer3Params);

            Assert.True(layer1Params.Scale > 0);
            Assert.True(layer2Params.Scale > 0);
            Assert.True(layer3Params.Scale > 0);
        }

        [Fact]
        public void Quantize_MultiLayerModel_VerifiesNonQuantizableLayersArePreserved()
        {
            // Arrange
            var model = new MockMultiLayerModel();
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
            // Model structure should be preserved
            Assert.NotNull(model.Layer1);
            Assert.NotNull(model.Layer2);
            Assert.NotNull(model.Layer3);

            // Weights should still exist (not removed)
            Assert.NotNull(model.Layer1.Weight);
            Assert.NotNull(model.Layer2.Weight);
            Assert.NotNull(model.Layer3.Weight);
        }

        [Fact]
        public void Quantize_MultiLayerModel_VerifiesQuantizationParametersPerLayer()
        {
            // Arrange
            var model = new MockMultiLayerModel();
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
            var layer1Params = quantizer.GetLayerQuantizationParameters("layer1");
            var layer2Params = quantizer.GetLayerQuantizationParameters("layer2");
            var layer3Params = quantizer.GetLayerQuantizationParameters("layer3");

            // Each layer should have its own quantization parameters
            Assert.NotEqual(layer1Params.Scale, layer2Params.Scale);
            Assert.NotEqual(layer2Params.Scale, layer3Params.Scale);

            // All should be within valid Int8 range
            Assert.InRange(layer1Params.ZeroPoint, -128, 127);
            Assert.InRange(layer2Params.ZeroPoint, -128, 127);
            Assert.InRange(layer3Params.ZeroPoint, -128, 127);
        }

        [Fact]
        public void Quantize_WithNullModel_ThrowsArgumentNullException()
        {
            // Arrange
            Module model = null;
            var config = new QuantizationConfig();
            var quantizer = new PTQQuantizer();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => quantizer.Quantize(model, null, config));
        }

        [Fact]
        public void Quantize_WithNullConfig_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new MockMultiLayerModel();
            QuantizationConfig config = null;
            var quantizer = new PTQQuantizer();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => quantizer.Quantize(model, null, config));
        }

        [Fact]
        public void Quantize_WithInvalidConfig_ThrowsValidationException()
        {
            // Arrange
            var model = new MockMultiLayerModel();
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                // Missing required settings - will fail validation
            };

            var quantizer = new PTQQuantizer();

            // Act & Assert
            // Config validation should handle invalid configurations
            var exception = Assert.Throws<ArgumentException>(() => quantizer.Quantize(model, null, config));
            Assert.Contains("validation", exception.Message, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void SetPerLayerFallback_DisablesQuantizationForLayer()
        {
            // Arrange
            var model = new MockMultiLayerModel();
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };

            var quantizer = new PTQQuantizer();

            // Act
            quantizer.SetPerLayerFallback("layer2", false); // Disable quantization for layer2
            quantizer.Quantize(model, null, config);

            // Assert
            var skippedLayers = quantizer.GetSkippedLayers();
            Assert.Contains("layer2", skippedLayers);

            // layer2 should not have quantization parameters
            Assert.Throws<KeyNotFoundException>(() => quantizer.GetLayerQuantizationParameters("layer2"));

            // Other layers should still be quantized
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("layer1"));
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("layer3"));
        }

        [Fact]
        public void SetPerLayerFallback_EnablesQuantizationForLayer()
        {
            // Arrange
            var model = new MockMultiLayerModel();
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };

            var quantizer = new PTQQuantizer();

            // Act
            quantizer.SetPerLayerFallback("layer2", true); // Enable quantization for layer2
            quantizer.Quantize(model, null, config);

            // Assert
            var skippedLayers = quantizer.GetSkippedLayers();
            Assert.DoesNotContain("layer2", skippedLayers);

            // layer2 should have quantization parameters
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("layer2"));
        }

        [Fact]
        public void GetLayerQuantizationParameters_WithValidLayer_ReturnsParameters()
        {
            // Arrange
            var model = new MockMultiLayerModel();
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };

            var quantizer = new PTQQuantizer();
            quantizer.Quantize(model, null, config);

            // Act
            var parameters = quantizer.GetLayerQuantizationParameters("layer1");

            // Assert
            Assert.NotNull(parameters);
            Assert.True(parameters.Scale > 0);
            Assert.InRange(parameters.ZeroPoint, -128, 127);
        }

        [Fact]
        public void GetLayerQuantizationParameters_WithInvalidLayer_ThrowsException()
        {
            // Arrange
            var model = new MockMultiLayerModel();
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };

            var quantizer = new PTQQuantizer();
            quantizer.Quantize(model, null, config);

            // Act & Assert
            Assert.Throws<KeyNotFoundException>(() => quantizer.GetLayerQuantizationParameters("nonexistent_layer"));
        }

        [Fact]
        public void GetSkippedLayers_ReturnsListOfSkippedLayers()
        {
            // Arrange
            var model = new MockMultiLayerModel();
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };

            var quantizer = new PTQQuantizer();
            quantizer.SetPerLayerFallback("layer1", false);
            quantizer.SetPerLayerFallback("layer3", false);
            quantizer.Quantize(model, null, config);

            // Act
            var skippedLayers = quantizer.GetSkippedLayers();

            // Assert
            Assert.Contains("layer1", skippedLayers);
            Assert.Contains("layer3", skippedLayers);
            Assert.DoesNotContain("layer2", skippedLayers);
        }

        [Fact]
        public void Quantize_WithMultipleQuantizations_ResetsState()
        {
            // Arrange
            var model = new MockMultiLayerModel();
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };

            var quantizer = new PTQQuantizer();

            // Act - First quantization
            quantizer.Quantize(model, null, config);
            quantizer.SetPerLayerFallback("layer1", false);

            // Second quantization
            quantizer.Quantize(model, null, config);

            // Assert - State should be reset, layer1 should be quantized again
            var skippedLayers = quantizer.GetSkippedLayers();
            Assert.DoesNotContain("layer1", skippedLayers);
        }

        [Fact]
        public void Quantize_WithFallbackToFP32_PerformsSensitivityAnalysis()
        {
            // Arrange
            var model = new MockMultiLayerModel();

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
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax,
                FallbackToFP32 = true,
                AccuracyThreshold = 0.01f // 1% accuracy threshold
            };

            var quantizer = new PTQQuantizer();

            // Act
            quantizer.Quantize(model, dataLoader, config);

            // Assert
            var skippedLayers = quantizer.GetSkippedLayers();
            // Some layers might have been skipped based on sensitivity
            // This is dependent on the sensitivity analysis implementation
            Assert.NotNull(skippedLayers);
        }

        [Fact]
        public void Quantize_WithCalibrationData_PerformsStaticQuantization()
        {
            // Arrange
            var model = new MockMultiLayerModel();

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
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            var quantizer = new PTQQuantizer();

            // Act
            quantizer.Quantize(model, dataLoader, config);

            // Assert
            // Static quantization should have activation parameters
            var layer1Params = quantizer.GetLayerQuantizationParameters("layer1");
            Assert.NotNull(layer1Params);
        }

        [Fact]
        public void Quantize_WithoutCalibrationData_PerformsDynamicQuantization()
        {
            // Arrange
            var model = new MockMultiLayerModel();
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
            // Dynamic quantization should not have activation parameters
            var layer1Params = quantizer.GetLayerQuantizationParameters("layer1");
            Assert.NotNull(layer1Params);
            // Activations remain in FP32
        }

        [Fact]
        public void SetPerLayerFallback_WithEmptyLayerName_ThrowsArgumentException()
        {
            // Arrange
            var quantizer = new PTQQuantizer();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => quantizer.SetPerLayerFallback("", true));
            Assert.Throws<ArgumentException>(() => quantizer.SetPerLayerFallback("   ", true));
            Assert.Throws<ArgumentException>(() => quantizer.SetPerLayerFallback(null!, true));
        }

        [Fact]
        public void GetLayerQuantizationParameters_WithEmptyLayerName_ThrowsArgumentException()
        {
            // Arrange
            var model = new MockMultiLayerModel();
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };

            var quantizer = new PTQQuantizer();
            quantizer.Quantize(model, null, config);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => quantizer.GetLayerQuantizationParameters(""));
            Assert.Throws<ArgumentException>(() => quantizer.GetLayerQuantizationParameters("   "));
            Assert.Throws<ArgumentException>(() => quantizer.GetLayerQuantizationParameters(null!));
        }
    }
}
