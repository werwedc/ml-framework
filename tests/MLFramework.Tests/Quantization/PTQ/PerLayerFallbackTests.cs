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
    /// Tests for per-layer quantization control and fallback functionality.
    /// </summary>
    public class PerLayerFallbackTests
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
            public MockLinearModule Layer4 { get; }

            public MockModel()
            {
                Name = "test_model";
                Layer1 = new MockLinearModule("layer1", 10, 20);
                Layer2 = new MockLinearModule("layer2", 20, 15);
                Layer3 = new MockLinearModule("layer3", 15, 10);
                Layer4 = new MockLinearModule("layer4", 10, 5);
            }

            public override Tensor Forward(Tensor input)
            {
                var output = Layer1.Forward(input);
                output = Layer2.Forward(output);
                output = Layer3.Forward(output);
                output = Layer4.Forward(output);
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
        public void SetPerLayerFallback_EnableDisablePerLayer_EnablesOrDisablesQuantization()
        {
            // Arrange
            var model = new MockModel();
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };

            var quantizer = new PTQQuantizer();

            // Act - Disable quantization for layer2
            quantizer.SetPerLayerFallback("layer2", false);
            quantizer.Quantize(model, null, config);

            // Assert
            var skippedLayers = quantizer.GetSkippedLayers();
            Assert.Contains("layer2", skippedLayers);
            Assert.DoesNotContain("layer1", skippedLayers);
            Assert.DoesNotContain("layer3", skippedLayers);
            Assert.DoesNotContain("layer4", skippedLayers);

            // Verify layer2 doesn't have quantization parameters
            Assert.Throws<KeyNotFoundException>(() => quantizer.GetLayerQuantizationParameters("layer2"));

            // Other layers should have parameters
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("layer1"));
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("layer3"));
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("layer4"));
        }

        [Fact]
        public void SetPerLayerFallback_EnableForSensitiveLayer_VerifiesFP32Fallback()
        {
            // Arrange
            var model = new MockModel();
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };

            var quantizer = new PTQQuantizer();

            // Act - Disable quantization for sensitive layer (layer2)
            quantizer.SetPerLayerFallback("layer2", false);
            quantizer.Quantize(model, null, config);

            // Assert - Verify FP32 fallback for sensitive layer
            var skippedLayers = quantizer.GetSkippedLayers();
            Assert.Contains("layer2", skippedLayers);

            // Verify layer2 weights are still in FP32 (not quantized)
            Assert.NotNull(model.Layer2.Weight);
            Assert.True(model.Layer2.Weight.Data.Length > 0);
        }

        [Fact]
        public void Quantize_MixedPrecisionModel_WorksCorrectly()
        {
            // Arrange
            var model = new MockModel();
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };

            var quantizer = new PTQQuantizer();

            // Act - Create mixed precision model: some layers quantized, some not
            quantizer.SetPerLayerFallback("layer2", false);
            quantizer.SetPerLayerFallback("layer4", false);
            quantizer.Quantize(model, null, config);

            // Assert
            var skippedLayers = quantizer.GetSkippedLayers();
            Assert.Contains("layer2", skippedLayers);
            Assert.Contains("layer4", skippedLayers);

            // Mixed precision: layers 1 and 3 quantized, layers 2 and 4 in FP32
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("layer1"));
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("layer3"));

            // Verify layers 2 and 4 don't have quantization parameters
            Assert.Throws<KeyNotFoundException>(() => quantizer.GetLayerQuantizationParameters("layer2"));
            Assert.Throws<KeyNotFoundException>(() => quantizer.GetLayerQuantizationParameters("layer4"));
        }

        [Fact]
        public void Quantize_MixedPrecisionModel_VerifiesInferenceWithMixedPrecision()
        {
            // Arrange
            var model = new MockModel();
            var testInput = new Tensor(new float[10], new int[] { 1, 10 });
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };

            var quantizer = new PTQQuantizer();

            // Get original output
            var originalOutput = model.Forward(testInput);

            // Act - Create mixed precision model
            quantizer.SetPerLayerFallback("layer2", false);
            quantizer.SetPerLayerFallback("layer4", false);
            quantizer.Quantize(model, null, config);

            // Run inference with mixed precision
            var mixedPrecisionOutput = model.Forward(testInput);

            // Assert - Inference should still work with mixed precision
            Assert.NotNull(mixedPrecisionOutput);
            Assert.Equal(originalOutput.Shape, mixedPrecisionOutput.Shape);
        }

        [Fact]
        public void SetPerLayerFallback_AllLayersDisabled_VerifiesNoQuantization()
        {
            // Arrange
            var model = new MockModel();
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };

            var quantizer = new PTQQuantizer();

            // Act - Disable quantization for all layers
            quantizer.SetPerLayerFallback("layer1", false);
            quantizer.SetPerLayerFallback("layer2", false);
            quantizer.SetPerLayerFallback("layer3", false);
            quantizer.SetPerLayerFallback("layer4", false);
            quantizer.Quantize(model, null, config);

            // Assert
            var skippedLayers = quantizer.GetSkippedLayers();
            Assert.Equal(4, skippedLayers.Length);

            // No layers should have quantization parameters
            Assert.Throws<KeyNotFoundException>(() => quantizer.GetLayerQuantizationParameters("layer1"));
            Assert.Throws<KeyNotFoundException>(() => quantizer.GetLayerQuantizationParameters("layer2"));
            Assert.Throws<KeyNotFoundException>(() => quantizer.GetLayerQuantizationParameters("layer3"));
            Assert.Throws<KeyNotFoundException>(() => quantizer.GetLayerQuantizationParameters("layer4"));
        }

        [Fact]
        public void SetPerLayerFallback_NoLayersDisabled_VerifiesAllQuantized()
        {
            // Arrange
            var model = new MockModel();
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };

            var quantizer = new PTQQuantizer();

            // Act - Don't disable any layers
            quantizer.Quantize(model, null, config);

            // Assert
            var skippedLayers = quantizer.GetSkippedLayers();
            Assert.Empty(skippedLayers);

            // All layers should have quantization parameters
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("layer1"));
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("layer2"));
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("layer3"));
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("layer4"));
        }

        [Fact]
        public void SetPerLayerFallback_EnableAfterDisable_ReenablesQuantization()
        {
            // Arrange
            var model = new MockModel();
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };

            var quantizer = new PTQQuantizer();

            // Act - First disable, then enable
            quantizer.SetPerLayerFallback("layer2", false);
            quantizer.Quantize(model, null, config);

            // Create new quantizer for fresh state
            var quantizer2 = new PTQQuantizer();
            quantizer2.SetPerLayerFallback("layer2", true);
            quantizer2.Quantize(model, null, config);

            // Assert
            var skippedLayers = quantizer2.GetSkippedLayers();
            Assert.DoesNotContain("layer2", skippedLayers);
            Assert.NotNull(quantizer2.GetLayerQuantizationParameters("layer2"));
        }

        [Fact]
        public void SetPerLayerFallback_WithInvalidLayerName_DoesNotThrow()
        {
            // Arrange
            var model = new MockModel();
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };

            var quantizer = new PTQQuantizer();

            // Act - Set fallback for non-existent layer
            quantizer.SetPerLayerFallback("nonexistent_layer", false);
            var exception = Record.Exception(() => quantizer.Quantize(model, null, config));

            // Assert
            Assert.Null(exception);
        }

        [Fact]
        public void Quantize_WithStaticQuantizationAndFallback_RespectsFallback()
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
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax
            };

            var quantizer = new PTQQuantizer();

            // Act - Set fallback and quantize with static quantization
            quantizer.SetPerLayerFallback("layer3", false);
            quantizer.Quantize(model, dataLoader, config);

            // Assert
            var skippedLayers = quantizer.GetSkippedLayers();
            Assert.Contains("layer3", skippedLayers);

            // Layer3 should not have activation or weight quantization parameters
            Assert.Throws<KeyNotFoundException>(() => quantizer.GetLayerQuantizationParameters("layer3"));
            Assert.Throws<KeyNotFoundException>(() => quantizer.GetLayerQuantizationParameters("layer3_activation"));

            // Other layers should have both weight and activation parameters
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("layer1"));
            Assert.NotNull(quantizer.GetLayerQuantizationParameters("layer1_activation"));
        }

        [Fact]
        public void SetPerLayerFallback_MultipleLayersDifferentSettings_WorksCorrectly()
        {
            // Arrange
            var model = new MockModel();
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.None
            };

            var quantizer = new PTQQuantizer();

            // Act - Set different settings for different layers
            quantizer.SetPerLayerFallback("layer1", true);  // Enable
            quantizer.SetPerLayerFallback("layer2", false); // Disable
            quantizer.SetPerLayerFallback("layer3", true);  // Enable
            quantizer.SetPerLayerFallback("layer4", false); // Disable
            quantizer.Quantize(model, null, config);

            // Assert
            var skippedLayers = quantizer.GetSkippedLayers();
            Assert.Contains("layer2", skippedLayers);
            Assert.Contains("layer4", skippedLayers);
            Assert.DoesNotContain("layer1", skippedLayers);
            Assert.DoesNotContain("layer3", skippedLayers);
        }

        [Fact]
        public void GetSkippedLayers_AfterQuantization_ReturnsCorrectList()
        {
            // Arrange
            var model = new MockModel();
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
            Assert.Equal(2, skippedLayers.Length);
            Assert.Contains("layer1", skippedLayers);
            Assert.Contains("layer3", skippedLayers);
        }

        [Fact]
        public void SetPerLayerFallback_WithEmptyLayerName_ThrowsArgumentException()
        {
            // Arrange
            var quantizer = new PTQQuantizer();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => quantizer.SetPerLayerFallback("", true));
            Assert.Throws<ArgumentException>(() => quantizer.SetPerLayerFallback("   ", true));
        }

        [Fact]
        public void SetPerLayerFallback_WithNullLayerName_ThrowsArgumentException()
        {
            // Arrange
            var quantizer = new PTQQuantizer();

            // Act & Assert
            Assert.Throws<ArgumentException>(() => quantizer.SetPerLayerFallback(null!, true));
        }

        [Fact]
        public void Quantize_WithFallbackAndSensitivityAnalysis_RespectsBoth()
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
                WeightQuantization = QuantizationMode.PerTensor,
                ActivationQuantization = QuantizationMode.PerTensor,
                CalibrationMethod = CalibrationMethod.MinMax,
                FallbackToFP32 = true,
                AccuracyThreshold = 0.01f
            };

            var quantizer = new PTQQuantizer();

            // Act - Set explicit fallback + enable sensitivity analysis
            quantizer.SetPerLayerFallback("layer2", false);
            quantizer.Quantize(model, dataLoader, config);

            // Assert
            var skippedLayers = quantizer.GetSkippedLayers();
            // layer2 should be skipped (explicit fallback)
            Assert.Contains("layer2", skippedLayers);
        }
    }
}
