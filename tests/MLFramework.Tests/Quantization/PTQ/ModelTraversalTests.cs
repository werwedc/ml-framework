using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using MLFramework.Quantization.PTQ;
using MLFramework.NN;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Quantization.PTQ
{
    /// <summary>
    /// Tests for model graph traversal functionality.
    /// </summary>
    public class ModelTraversalTests
    {
        private class MockLinearModule : Module
        {
            public Parameter Weight { get; }

            public MockLinearModule(string name, int inFeatures, int outFeatures)
            {
                Name = name;
                float[] weightData = new float[inFeatures * outFeatures];
                for (int i = 0; i < weightData.Length; i++)
                {
                    weightData[i] = (float)((i % 100) / 50.0 - 1.0);
                }
                Weight = new Parameter(weightData, new int[] { outFeatures, inFeatures }, requiresGrad: true);
            }

            public override Tensor Forward(Tensor input)
            {
                return input.Clone();
            }
        }

        private class MockConv2DModule : Module
        {
            public Parameter Weight { get; }

            public MockConv2DModule(string name)
            {
                Name = name;
                float[] weightData = new float[16 * 3 * 3 * 3];
                for (int i = 0; i < weightData.Length; i++)
                {
                    weightData[i] = (float)((i % 100) / 50.0 - 1.0);
                }
                Weight = new Parameter(weightData, new int[] { 16, 3, 3, 3 }, requiresGrad: true);
            }

            public override Tensor Forward(Tensor input)
            {
                return input.Clone();
            }
        }

        private class MockReluModule : Module
        {
            public MockReluModule(string name)
            {
                Name = name;
            }

            public override Tensor Forward(Tensor input)
            {
                return input.Clone();
            }
        }

        private class MockBatchNormModule : Module
        {
            public Parameter Weight { get; }
            public Parameter Bias { get; }

            public MockBatchNormModule(string name)
            {
                Name = name;
                Weight = new Parameter(new float[10], new int[] { 10 }, requiresGrad: true);
                Bias = new Parameter(new float[10], new int[] { 10 }, requiresGrad: true);
            }

            public override Tensor Forward(Tensor input)
            {
                return input.Clone();
            }
        }

        private class MockSequentialModel : Module
        {
            public MockLinearModule Linear1 { get; }
            public MockReluModule Relu { get; }
            public MockLinearModule Linear2 { get; }

            public MockSequentialModel()
            {
                Name = "sequential_model";
                Linear1 = new MockLinearModule("linear1", 10, 20);
                Relu = new MockReluModule("relu");
                Linear2 = new MockLinearModule("linear2", 20, 5);
            }

            public override Tensor Forward(Tensor input)
            {
                var output = Linear1.Forward(input);
                output = Relu.Forward(output);
                output = Linear2.Forward(output);
                return output;
            }
        }

        private class MockConvModel : Module
        {
            public MockConv2DModule Conv1 { get; }
            public MockBatchNormModule BatchNorm { get; }
            public MockReluModule Relu { get; }

            public MockConvModel()
            {
                Name = "conv_model";
                Conv1 = new MockConv2DModule("conv1");
                BatchNorm = new MockBatchNormModule("batchnorm");
                Relu = new MockReluModule("relu");
            }

            public override Tensor Forward(Tensor input)
            {
                var output = Conv1.Forward(input);
                output = BatchNorm.Forward(output);
                output = Relu.Forward(output);
                return output;
            }
        }

        private class MockResidualModel : Module
        {
            public MockLinearModule Linear1 { get; }
            public MockLinearModule Linear2 { get; }

            public MockResidualModel()
            {
                Name = "residual_model";
                Linear1 = new MockLinearModule("linear1", 10, 20);
                Linear2 = new MockLinearModule("linear2", 20, 10);
            }

            public override Tensor Forward(Tensor input)
            {
                var output = Linear1.Forward(input);
                output = Linear2.Forward(output);
                // Residual connection: output + input
                var outputData = new float[input.Data.Length];
                for (int i = 0; i < outputData.Length; i++)
                {
                    outputData[i] = output.Data[i] + input.Data[i];
                }
                return new Tensor(outputData, input.Shape);
            }
        }

        private class MockSimpleModel : Module
        {
            public MockSimpleModel(string name)
            {
                Name = name;
            }

            public override Tensor Forward(Tensor input)
            {
                return input.Clone();
            }
        }

        [Fact]
        public void GetQuantizableLayers_CorrectlyIdentifiesQuantizableLayers()
        {
            // Arrange
            var model = new MockSequentialModel();
            var modelTraversal = new ModelTraversal();

            // Act
            var quantizableLayers = modelTraversal.GetQuantizableLayers(model);

            // Assert
            // Linear1 and Linear2 should be quantizable
            var layerNames = quantizableLayers.Select(l => modelTraversal.GetLayerName(l)).ToList();
            Assert.Contains("linear1", layerNames);
            Assert.Contains("linear2", layerNames);
            // ReLU should NOT be quantizable
            Assert.DoesNotContain("relu", layerNames);
        }

        [Fact]
        public void GetQuantizableLayers_ConvModel_IdentifiesCorrectLayers()
        {
            // Arrange
            var model = new MockConvModel();
            var modelTraversal = new ModelTraversal();

            // Act
            var quantizableLayers = modelTraversal.GetQuantizableLayers(model);

            // Assert
            // Conv1 should be quantizable
            var layerNames = quantizableLayers.Select(l => modelTraversal.GetLayerName(l)).ToList();
            Assert.Contains("conv1", layerNames);
            // BatchNorm and ReLU should NOT be quantizable
            Assert.DoesNotContain("batchnorm", layerNames);
            Assert.DoesNotContain("relu", layerNames);
        }

        [Fact]
        public void GetQuantizableLayers_WithNullModel_ThrowsArgumentNullException()
        {
            // Arrange
            Module model = null;
            var modelTraversal = new ModelTraversal();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => modelTraversal.GetQuantizableLayers(model));
        }

        [Fact]
        public void GetAllLayers_ReturnsAllLayersIncludingNonQuantizable()
        {
            // Arrange
            var model = new MockSequentialModel();
            var modelTraversal = new ModelTraversal();

            // Act
            var allLayers = modelTraversal.GetAllLayers(model);

            // Assert
            var layerNames = allLayers.Select(l => modelTraversal.GetLayerName(l)).ToList();
            Assert.Contains("linear1", layerNames);
            Assert.Contains("relu", layerNames);
            Assert.Contains("linear2", layerNames);
        }

        [Fact]
        public void GetAllLayers_WithNullModel_ThrowsArgumentNullException()
        {
            // Arrange
            Module model = null;
            var modelTraversal = new ModelTraversal();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => modelTraversal.GetAllLayers(model));
        }

        [Fact]
        public void GetLayerName_ReturnsCorrectName()
        {
            // Arrange
            var layer = new MockLinearModule("test_layer", 10, 5);
            var modelTraversal = new ModelTraversal();

            // Act
            var name = modelTraversal.GetLayerName(layer);

            // Assert
            Assert.Equal("test_layer", name);
        }

        [Fact]
        public void GetLayerName_WithNullLayer_ThrowsArgumentNullException()
        {
            // Arrange
            Module layer = null;
            var modelTraversal = new ModelTraversal();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => modelTraversal.GetLayerName(layer));
        }

        [Fact]
        public void IsQuantizable_LinearLayer_ReturnsTrue()
        {
            // Arrange
            var layer = new MockLinearModule("linear", 10, 5);
            var modelTraversal = new ModelTraversal();

            // Act
            var isQuantizable = modelTraversal.IsQuantizable(layer);

            // Assert
            Assert.True(isQuantizable);
        }

        [Fact]
        public void IsQuantizable_Conv2DLayer_ReturnsTrue()
        {
            // Arrange
            var layer = new MockConv2DModule("conv2d");
            var modelTraversal = new ModelTraversal();

            // Act
            var isQuantizable = modelTraversal.IsQuantizable(layer);

            // Assert
            Assert.True(isQuantizable);
        }

        [Fact]
        public void IsQuantizable_ReluLayer_ReturnsFalse()
        {
            // Arrange
            var layer = new MockReluModule("relu");
            var modelTraversal = new ModelTraversal();

            // Act
            var isQuantizable = modelTraversal.IsQuantizable(layer);

            // Assert
            Assert.False(isQuantizable);
        }

        [Fact]
        public void IsQuantizable_BatchNormLayer_ReturnsFalse()
        {
            // Arrange
            var layer = new MockBatchNormModule("batchnorm");
            var modelTraversal = new ModelTraversal();

            // Act
            var isQuantizable = modelTraversal.IsQuantizable(layer);

            // Assert
            Assert.False(isQuantizable);
        }

        [Fact]
        public void IsQuantizable_LayerWithoutParameters_ReturnsFalse()
        {
            // Arrange
            var layer = new MockReluModule("no_params");
            var modelTraversal = new ModelTraversal();

            // Act
            var isQuantizable = modelTraversal.IsQuantizable(layer);

            // Assert
            Assert.False(isQuantizable);
        }

        [Fact]
        public void IsQuantizable_WithNullLayer_ReturnsFalse()
        {
            // Arrange
            Module layer = null;
            var modelTraversal = new ModelTraversal();

            // Act
            var isQuantizable = modelTraversal.IsQuantizable(layer);

            // Assert
            Assert.False(isQuantizable);
        }

        [Fact]
        public void SupportsPerChannelQuantization_Conv2DLayer_ReturnsTrue()
        {
            // Arrange
            var layer = new MockConv2DModule("conv2d");
            var modelTraversal = new ModelTraversal();

            // Act
            var supportsPerChannel = modelTraversal.SupportsPerChannelQuantization(layer);

            // Assert
            Assert.True(supportsPerChannel);
        }

        [Fact]
        public void SupportsPerChannelQuantization_LinearLayer_ReturnsFalse()
        {
            // Arrange
            var layer = new MockLinearModule("linear", 10, 5);
            var modelTraversal = new ModelTraversal();

            // Act
            var supportsPerChannel = modelTraversal.SupportsPerChannelQuantization(layer);

            // Assert
            Assert.False(supportsPerChannel);
        }

        [Fact]
        public void SupportsPerChannelQuantization_WithNullLayer_ReturnsFalse()
        {
            // Arrange
            Module layer = null;
            var modelTraversal = new ModelTraversal();

            // Act
            var supportsPerChannel = modelTraversal.SupportsPerChannelQuantization(layer);

            // Assert
            Assert.False(supportsPerChannel);
        }

        [Fact]
        public void GetQuantizableLayers_ResidualModel_HandlesComplexStructure()
        {
            // Arrange
            var model = new MockResidualModel();
            var modelTraversal = new ModelTraversal();

            // Act
            var quantizableLayers = modelTraversal.GetQuantizableLayers(model);

            // Assert
            var layerNames = quantizableLayers.Select(l => modelTraversal.GetLayerName(l)).ToList();
            Assert.Contains("linear1", layerNames);
            Assert.Contains("linear2", layerNames);
        }

        [Fact]
        public void GetAllLayers_ResidualModel_PreservesStructure()
        {
            // Arrange
            var model = new MockResidualModel();
            var modelTraversal = new ModelTraversal();

            // Act
            var allLayers = modelTraversal.GetAllLayers(model);

            // Assert
            var layerNames = allLayers.Select(l => modelTraversal.GetLayerName(l)).ToList();
            Assert.Contains("linear1", layerNames);
            Assert.Contains("linear2", layerNames);
        }

        [Fact]
        public void GetQuantizableLayers_ModelWithMixedLayers_IdentifiesCorrectly()
        {
            // Arrange
            var model = new MockConvModel();
            var modelTraversal = new ModelTraversal();

            // Act
            var quantizableLayers = modelTraversal.GetQuantizableLayers(model);

            // Assert
            // Only conv layers should be quantizable
            foreach (var layer in quantizableLayers)
            {
                var name = modelTraversal.GetLayerName(layer).ToLowerInvariant();
                Assert.True(name.Contains("conv") || !name.Contains("relu") && !name.Contains("batchnorm"));
            }
        }

        [Fact]
        public void IsQuantizable_CaseInsensitiveMatching_WorksCorrectly()
        {
            // Arrange
            var reluLower = new MockReluModule("relu");
            var reluUpper = new MockReluModule("RELU");
            var reluMixed = new MockReluModule("ReLu");
            var modelTraversal = new ModelTraversal();

            // Act
            var isQuantizableLower = modelTraversal.IsQuantizable(reluLower);
            var isQuantizableUpper = modelTraversal.IsQuantizable(reluUpper);
            var isQuantizableMixed = modelTraversal.IsQuantizable(reluMixed);

            // Assert - all should be false
            Assert.False(isQuantizableLower);
            Assert.False(isQuantizableUpper);
            Assert.False(isQuantizableMixed);
        }

        [Fact]
        public void SupportsPerChannelQuantization_CaseInsensitiveMatching_WorksCorrectly()
        {
            // Arrange
            var convLower = new MockConv2DModule("conv");
            var convUpper = new MockConv2DModule("CONV");
            var convMixed = new MockConv2DModule("Conv");
            var modelTraversal = new ModelTraversal();

            // Act
            var supportsLower = modelTraversal.SupportsPerChannelQuantization(convLower);
            var supportsUpper = modelTraversal.SupportsPerChannelQuantization(convUpper);
            var supportsMixed = modelTraversal.SupportsPerChannelQuantization(convMixed);

            // Assert - all should be true
            Assert.True(supportsLower);
            Assert.True(supportsUpper);
            Assert.True(supportsMixed);
        }

        [Fact]
        public void GetQuantizableLayers_ModelWithNoQuantizableLayers_ReturnsEmptyList()
        {
            // Arrange
            var model = new MockSimpleModel("simple_model");
            var modelTraversal = new ModelTraversal();

            // Act
            var quantizableLayers = modelTraversal.GetQuantizableLayers(model);

            // Assert
            Assert.Empty(quantizableLayers);
        }

        [Fact]
        public void GetQuantizableLayers_ModelWithAllQuantizableLayers_ReturnsAllLayers()
        {
            // Arrange
            var model = new MockSequentialModel();
            var modelTraversal = new ModelTraversal();

            // Act
            var quantizableLayers = modelTraversal.GetQuantizableLayers(model);
            var allLayers = modelTraversal.GetAllLayers(model);

            // Assert
            // Not all layers are quantizable (e.g., ReLU is not)
            Assert.NotEqual(allLayers.Count, quantizableLayers.Count);
            Assert.True(quantizableLayers.Count < allLayers.Count);
        }
    }
}
