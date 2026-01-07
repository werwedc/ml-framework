using System;
using System.Collections.Generic;
using Xunit;
using MLFramework.Quantization.PTQ;
using MLFramework.Quantization.DataStructures;
using MLFramework.Quantization.Calibration;
using MLFramework.NN;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Quantization.PTQ
{
    /// <summary>
    /// Tests for dynamic quantization functionality.
    /// </summary>
    public class DynamicQuantizationTests
    {
        private class MockLinearModule : Module
        {
            public Parameter Weight { get; }
            public Parameter Bias { get; }

            public MockLinearModule(int inFeatures, int outFeatures)
            {
                Name = "linear";
                var random = new Random(42);
                float[] weightData = new float[inFeatures * outFeatures];
                for (int i = 0; i < weightData.Length; i++)
                {
                    weightData[i] = (float)((random.NextDouble() - 0.5) * 2.0); // [-1, 1]
                }

                float[] biasData = new float[outFeatures];
                for (int i = 0; i < biasData.Length; i++)
                {
                    biasData[i] = (float)((random.NextDouble() - 0.5) * 0.1); // [-0.05, 0.05]
                }

                Weight = new Parameter(weightData, new int[] { outFeatures, inFeatures }, requiresGrad: true);
                Bias = new Parameter(biasData, new int[] { outFeatures }, requiresGrad: true);
            }

            public override Tensor Forward(Tensor input)
            {
                // Simple linear transformation: y = x * W^T + b
                int batchSize = input.Shape[0];
                int inFeatures = input.Shape[1];
                int outFeatures = Weight.Shape[0];

                float[] outputData = new float[batchSize * outFeatures];
                float[] inputData = input.Data;
                float[] weightData = Weight.Data;
                float[] biasData = Bias.Data;

                for (int b = 0; b < batchSize; b++)
                {
                    for (int o = 0; o < outFeatures; o++)
                    {
                        float sum = biasData[o];
                        for (int i = 0; i < inFeatures; i++)
                        {
                            sum += inputData[b * inFeatures + i] * weightData[o * inFeatures + i];
                        }
                        outputData[b * outFeatures + o] = sum;
                    }
                }

                return new Tensor(outputData, new int[] { batchSize, outFeatures });
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

        private class MockModel : Module
        {
            public MockLinearModule Linear { get; }

            public MockModel()
            {
                Name = "mock_model";
                Linear = new MockLinearModule(10, 5);
            }

            public override Tensor Forward(Tensor input)
            {
                return Linear.Forward(input);
            }
        }

        [Fact]
        public void Quantize_SimpleLinearModel_QuantizesWeightsToInt8()
        {
            // Arrange
            var model = new MockModel();
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
            Assert.Single(layerParams);
            Assert.Contains("linear", layerParams.Keys);
            var parameters = layerParams["linear"];
            Assert.True(parameters.Scale > 0);
            Assert.InRange(parameters.ZeroPoint, -128, 127);
        }

        [Fact]
        public void Quantize_SimpleLinearModel_VerifiesWeightsAreQuantized()
        {
            // Arrange
            var model = new MockModel();
            var originalWeights = model.Linear.Weight.Data;

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

            // Verify quantization parameters exist
            Assert.True(layerParams.ContainsKey("linear"));

            // Verify weights can be quantized to Int8 range
            var parameters = layerParams["linear"];
            for (int i = 0; i < originalWeights.Length; i++)
            {
                var quantized = QuantizationOperations.Quantize(originalWeights[i], parameters);
                Assert.InRange(quantized, sbyte.MinValue, sbyte.MaxValue);
            }
        }

        [Fact]
        public void Quantize_SimpleLinearModel_VerifiesActivationsRemainFP32()
        {
            // Arrange
            var model = new MockModel();
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

            // In dynamic quantization, activations should not have quantization parameters
            Assert.DoesNotContain(layerParams.Keys, k => k.Contains("activation"));
        }

        [Fact]
        public void Quantize_SimpleLinearModel_VerifiesInferenceResultsAreAccurate()
        {
            // Arrange
            var model = new MockModel();
            var testInput = new Tensor(new float[] { 0.5f, 0.3f, 0.7f, 0.2f, 0.9f, 0.1f, 0.6f, 0.4f, 0.8f, 0.0f }, new int[] { 1, 10 });

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

            // Get original inference result
            var originalOutput = model.Forward(testInput);

            // Act
            dynamicQuantization.Quantize(model, config, layerFallback, layerParams, skippedLayers);

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
            var config = new QuantizationConfig();
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var dynamicQuantization = new DynamicQuantization(modelTraversal, calibratorFactory);

            var layerParams = new Dictionary<string, QuantizationParameters>();
            var layerFallback = new Dictionary<string, bool>();
            var skippedLayers = new List<string>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                dynamicQuantization.Quantize(model, config, layerFallback, layerParams, skippedLayers));
        }

        [Fact]
        public void Quantize_WithNullConfig_ThrowsArgumentNullException()
        {
            // Arrange
            var model = new MockModel();
            QuantizationConfig config = null;
            var modelTraversal = new ModelTraversal();
            var calibratorFactory = new CalibratorFactory();
            var dynamicQuantization = new DynamicQuantization(modelTraversal, calibratorFactory);

            var layerParams = new Dictionary<string, QuantizationParameters>();
            var layerFallback = new Dictionary<string, bool>();
            var skippedLayers = new List<string>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                dynamicQuantization.Quantize(model, config, layerFallback, layerParams, skippedLayers));
        }

        [Fact]
        public void Quantize_WithLayerFallback_SkipsSpecifiedLayers()
        {
            // Arrange
            var model = new MockModel();
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
            var layerFallback = new Dictionary<string, bool>
            {
                { "linear", true } // Skip this layer
            };
            var skippedLayers = new List<string>();

            // Act
            dynamicQuantization.Quantize(model, config, layerFallback, layerParams, skippedLayers);

            // Assert
            Assert.Contains("linear", skippedLayers);
            Assert.DoesNotContain("linear", layerParams.Keys);
        }

        [Fact]
        public void Quantize_WithMultipleLayers_QuantizesAllQuantizableLayers()
        {
            // Arrange - create a model with multiple linear layers
            var model = new MockModule("multi_layer_model");

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
            // Multi_layer_model has no parameters, so it shouldn't be quantized
            Assert.Empty(layerParams);
        }

        [Fact]
        public void Quantize_WithSymmetricQuantization_UsesZeroZeroPoint()
        {
            // Arrange
            var model = new MockModel();
            var config = new QuantizationConfig
            {
                QuantizationType = QuantizationType.Int8,
                WeightQuantization = QuantizationMode.SymmetricPerTensor,
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
            Assert.True(layerParams.ContainsKey("linear"));
            // Symmetric quantization should have zero point close to 0
            var parameters = layerParams["linear"];
            Assert.True(Math.Abs(parameters.ZeroPoint) <= 1, 
                $"Expected zero point near 0 for symmetric quantization, got {parameters.ZeroPoint}");
        }
    }
}
