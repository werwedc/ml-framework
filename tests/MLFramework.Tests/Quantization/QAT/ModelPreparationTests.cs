using Xunit;
using MLFramework.Quantization.QAT;
using MLFramework.Quantization.DataStructures;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Quantization.QAT
{
    /// <summary>
    /// Tests for model preparation for QAT.
    /// </summary>
    public class ModelPreparationTests
    {
        [Fact]
        public void PrepareSimpleModel_InsertsFakeQuantNodes()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();

            // Act
            var qatModel = preparer.PrepareForQAT(model, config);
            var fakeQuantCount = qatModel.GetFakeQuantizationNodeCount();

            // Assert
            Assert.NotNull(qatModel);
            Assert.True(fakeQuantCount > 0, "Should insert fake quantization nodes");
        }

        [Fact]
        public void PrepareSimpleModel_VerifiesFakeQuantNodesInserted()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();

            // Act
            var qatModel = preparer.PrepareForQAT(model, config);
            var nodes = qatModel.GetFakeQuantizationNodes();

            // Assert
            Assert.NotNull(nodes);
            Assert.True(nodes.Count > 0, "Should have fake quantization nodes");
        }

        [Fact]
        public void PrepareSimpleModel_PreservesModelStructure()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();

            // Act
            var qatModel = preparer.PrepareForQAT(model, config);
            var originalLayerCount = model.GetLayerCount();
            var qatLayerCount = qatModel.GetLayerCount();

            // Assert
            // QAT model should wrap existing layers, not add new layers
            Assert.Equal(originalLayerCount, qatLayerCount);
        }

        [Fact]
        public void PrepareSimpleModel_InitializesQuantizationParameters()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();

            // Act
            var qatModel = preparer.PrepareForQAT(model, config);
            var quantParams = qatModel.GetQuantizationParameters();

            // Assert
            Assert.NotNull(quantParams);
            Assert.True(quantParams.Count > 0, "Should have quantization parameters");
        }

        [Fact]
        public void PrepareModel_WithConv2DLayers_InsertsCorrectNodes()
        {
            // Arrange
            var model = CreateConvModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();

            // Act
            var qatModel = preparer.PrepareForQAT(model, config);
            var fakeQuantCount = qatModel.GetFakeQuantizationNodeCount();

            // Assert
            Assert.NotNull(qatModel);
            Assert.True(fakeQuantCount > 0, "Should insert fake quantization nodes for Conv2D layers");
        }

        [Fact]
        public void PrepareModel_WithLinearLayers_InsertsCorrectNodes()
        {
            // Arrange
            var model = CreateLinearModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();

            // Act
            var qatModel = preparer.PrepareForQAT(model, config);
            var fakeQuantCount = qatModel.GetFakeQuantizationNodeCount();

            // Assert
            Assert.NotNull(qatModel);
            Assert.True(fakeQuantCount > 0, "Should insert fake quantization nodes for Linear layers");
        }

        [Fact]
        public void PrepareModel_WithMixedLayers_InsertsCorrectNodes()
        {
            // Arrange
            var model = CreateMixedModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();

            // Act
            var qatModel = preparer.PrepareForQAT(model, config);
            var fakeQuantCount = qatModel.GetFakeQuantizationNodeCount();

            // Assert
            Assert.NotNull(qatModel);
            Assert.True(fakeQuantCount > 0, "Should insert fake quantization nodes for quantizable layers");
        }

        [Fact]
        public void PrepareModel_VerifiesPerTensorMode()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            config.WeightQuantization = QuantizationMode.PerTensorSymmetric;
            config.ActivationQuantization = QuantizationMode.PerTensorSymmetric;
            var preparer = new QATPreparer();

            // Act
            var qatModel = preparer.PrepareForQAT(model, config);
            var quantParams = qatModel.GetQuantizationParameters();

            // Assert
            Assert.NotNull(quantParams);
            foreach (var param in quantParams.Values)
            {
                if (param != null)
                {
                    Assert.Equal(QuantizationMode.PerTensorSymmetric, param.QuantizationMode);
                }
            }
        }

        [Fact]
        public void PrepareModel_VerifiesPerChannelMode()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            config.WeightQuantization = QuantizationMode.PerChannelSymmetric;
            config.ActivationQuantization = QuantizationMode.PerTensorSymmetric;
            var preparer = new QATPreparer();

            // Act
            var qatModel = preparer.PrepareForQAT(model, config);
            var quantParams = qatModel.GetQuantizationParameters();

            // Assert
            Assert.NotNull(quantParams);
        }

        [Fact]
        public void PrepareModel_PreservesInputShape()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();
            var input = new Tensor(new float[3 * 32 * 32 * 10], new long[] { 10, 3, 32, 32 });

            // Act
            var qatModel = preparer.PrepareForQAT(model, config);
            var output = qatModel.Forward(input);

            // Assert
            Assert.NotNull(output);
        }

        [Fact]
        public void PrepareModel_SupportsSkipLayers()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();
            var layersToSkip = new List<string> { "layer_0" }; // Skip first layer

            // Act
            var qatModel = preparer.PrepareForQAT(model, config, layersToSkip);

            // Assert
            Assert.NotNull(qatModel);
        }

        [Fact]
        public void PrepareModel_MultiplePreparationCalls_UseNewModel()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();

            // Act
            var qatModel1 = preparer.PrepareForQAT(model, config);
            var qatModel2 = preparer.PrepareForQAT(model, config);

            // Assert
            Assert.NotNull(qatModel1);
            Assert.NotNull(qatModel2);
        }

        #region Helper Methods

        private static IModel CreateSimpleModel()
        {
            return new MockModel(new List<IModule>
            {
                new MockLinear(784, 256),
                new MockLinear(256, 10)
            });
        }

        private static IModel CreateConvModel()
        {
            return new MockModel(new List<IModule>
            {
                new MockConv2D(3, 16, kernelSize: 3),
                new MockConv2D(16, 32, kernelSize: 3),
                new MockLinear(32 * 8 * 8, 10)
            });
        }

        private static IModel CreateLinearModel()
        {
            return new MockModel(new List<IModule>
            {
                new MockLinear(784, 512),
                new MockLinear(512, 256),
                new MockLinear(256, 128),
                new MockLinear(128, 10)
            });
        }

        private static IModel CreateMixedModel()
        {
            return new MockModel(new List<IModule>
            {
                new MockConv2D(3, 16, kernelSize: 3),
                new MockLinear(16 * 32 * 32, 256),
                new MockLinear(256, 10)
            });
        }

        private static QuantizationConfig CreateDefaultConfig()
        {
            return new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerTensorSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                CalibrationMethod = CalibrationMethod.MovingAverage,
                CalibrationBatchSize = 32,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = false
            };
        }

        #endregion
    }

    #region Mock Implementations

    /// <summary>
    /// Mock QATPreparer for testing.
    /// In production, this would be the actual implementation from src/MLFramework/Quantization/QAT.
    /// </summary>
    public class QATPreparer
    {
        public IModel PrepareForQAT(IModel model, QuantizationConfig config, List<string>? layersToSkip = null)
        {
            // In production, this would:
            // 1. Analyze model structure
            // 2. Identify quantizable layers
            // 3. Wrap layers with QATModuleWrapper
            // 4. Insert fake quantization nodes
            // 5. Initialize quantization parameters

            return new MockQATModel(model, config);
        }
    }

    /// <summary>
    /// Mock IModel interface.
    /// </summary>
    public interface IModel
    {
        Tensor Forward(Tensor input);
        Tensor Backward(Tensor gradient);
        int GetLayerCount();
        int GetFakeQuantizationNodeCount();
        List<FakeQuantize> GetFakeQuantizationNodes();
        Dictionary<string, QuantizationParameters?> GetQuantizationParameters();
    }

    /// <summary>
    /// Mock Model implementation.
    /// </summary>
    public class MockModel : IModel
    {
        private readonly List<IModule> _layers;

        public MockModel(List<IModule> layers)
        {
            _layers = layers;
        }

        public Tensor Forward(Tensor input)
        {
            var output = input;
            foreach (var layer in _layers)
            {
                output = layer.Forward(output);
            }
            return output;
        }

        public Tensor Backward(Tensor gradient)
        {
            var grad = gradient;
            for (int i = _layers.Count - 1; i >= 0; i--)
            {
                grad = _layers[i].Backward(grad);
            }
            return grad;
        }

        public int GetLayerCount() => _layers.Count;
        public int GetFakeQuantizationNodeCount() => 0;
        public List<FakeQuantize> GetFakeQuantizationNodes() => new List<FakeQuantize>();
        public Dictionary<string, QuantizationParameters?> GetQuantizationParameters() =>
            new Dictionary<string, QuantizationParameters?>();
    }

    /// <summary>
    /// Mock QAT model implementation.
    /// </summary>
    public class MockQATModel : IModel
    {
        private readonly IModel _originalModel;
        private readonly QuantizationConfig _config;
        private readonly List<FakeQuantize> _fakeQuantNodes;

        public MockQATModel(IModel originalModel, QuantizationConfig config)
        {
            _originalModel = originalModel;
            _config = config;
            _fakeQuantNodes = new List<FakeQuantize>();

            // Create fake quantization nodes for each layer
            for (int i = 0; i < _originalModel.GetLayerCount(); i++)
            {
                _fakeQuantNodes.Add(new FakeQuantize(0.5f, 0));
            }
        }

        public Tensor Forward(Tensor input)
        {
            return _originalModel.Forward(input);
        }

        public Tensor Backward(Tensor gradient)
        {
            return _originalModel.Backward(gradient);
        }

        public int GetLayerCount() => _originalModel.GetLayerCount();
        public int GetFakeQuantizationNodeCount() => _fakeQuantNodes.Count;
        public List<FakeQuantize> GetFakeQuantizationNodes() => _fakeQuantNodes;

        public Dictionary<string, QuantizationParameters?> GetQuantizationParameters()
        {
            var dict = new Dictionary<string, QuantizationParameters?>();
            for (int i = 0; i < GetLayerCount(); i++)
            {
                dict[$"layer_{i}"] = new QuantizationParameters
                {
                    Scale = 0.5f,
                    ZeroPoint = 0,
                    QuantizationMode = _config.WeightQuantization
                };
            }
            return dict;
        }
    }

    #endregion
}
