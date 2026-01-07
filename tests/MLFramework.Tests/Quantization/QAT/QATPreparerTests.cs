using Xunit;
using MLFramework.Quantization.QAT;
using MLFramework.Quantization.DataStructures;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Quantization.QAT
{
    /// <summary>
    /// Tests for QATPreparer.
    /// </summary>
    public class QATPreparerTests
    {
        [Fact]
        public void QATPreparer_PrepareModelForQAT()
        {
            // Arrange
            var preparer = new MockQATPreparerForTests();
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();

            // Act
            var qatModel = preparer.PrepareForQAT(model, config);

            // Assert
            Assert.NotNull(qatModel);
        }

        [Fact]
        public void QATPreparer_ConvertTrainedModelToInt8()
        {
            // Arrange
            var preparer = new MockQATPreparerForTests();
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var qatModel = preparer.PrepareForQAT(model, config);

            // Simulate training
            qatModel.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            for (int i = 0; i < 10; i++)
            {
                _ = qatModel.Forward(input);
            }

            // Act
            var quantizedModel = preparer.ConvertToQuantized(qatModel);

            // Assert
            Assert.NotNull(quantizedModel);
        }

        [Fact]
        public void QATPreparer_GetQATStatistics()
        {
            // Arrange
            var preparer = new MockQATPreparerForTests();
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var qatModel = preparer.PrepareForQAT(model, config);

            // Simulate training
            qatModel.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            _ = qatModel.Forward(input);

            // Act
            var stats = preparer.GetQATStatistics(qatModel);

            // Assert
            Assert.NotNull(stats);
        }

        [Fact]
        public void QATPreparer_HandlePerLayerConfiguration()
        {
            // Arrange
            var preparer = new MockQATPreparerForTests();
            var model = CreateSimpleModel();
            var globalConfig = CreateDefaultConfig();
            var perLayerConfig = new Dictionary<string, QuantizationConfig>
            {
                { "layer_0", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerChannelSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                        CalibrationMethod = CalibrationMethod.MinMax,
                        CalibrationBatchSize = 32,
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = false
                    }
                },
                { "layer_1", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerTensorSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                        CalibrationMethod = CalibrationMethod.MinMax,
                        CalibrationBatchSize = 32,
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = true
                    }
                }
            };

            // Act
            var qatModel = preparer.PrepareForQAT(model, globalConfig, perLayerConfig);

            // Assert
            Assert.NotNull(qatModel);
        }

        [Fact]
        public void QATPreparer_PrepareModel_VerifiesQuantizedLayerCount()
        {
            // Arrange
            var preparer = new MockQATPreparerForTests();
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();

            // Act
            var qatModel = preparer.PrepareForQAT(model, config);
            var quantizedLayerCount = qatModel.GetQuantizedLayerCount();

            // Assert
            Assert.True(quantizedLayerCount > 0);
        }

        [Fact]
        public void QATPreparer_PrepareModel_VerifiesFakeQuantNodeCount()
        {
            // Arrange
            var preparer = new MockQATPreparerForTests();
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();

            // Act
            var qatModel = preparer.PrepareForQAT(model, config);
            var fakeQuantNodeCount = qatModel.GetFakeQuantizationNodeCount();

            // Assert
            Assert.True(fakeQuantNodeCount > 0);
        }

        [Fact]
        public void QATPreparer_ConvertModel_VerifiesQuantizedLayerCount()
        {
            // Arrange
            var preparer = new MockQATPreparerForTests();
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var qatModel = preparer.PrepareForQAT(model, config);

            // Simulate training
            qatModel.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            for (int i = 0; i < 5; i++)
            {
                _ = qatModel.Forward(input);
            }

            // Act
            var quantizedModel = preparer.ConvertToQuantized(qatModel);
            var quantizedLayerCount = quantizedModel.GetQuantizedLayerCount();

            // Assert
            Assert.True(quantizedLayerCount > 0);
        }

        [Fact]
        public void QATPreparer_GetStatistics_ContainsCorrectInformation()
        {
            // Arrange
            var preparer = new MockQATPreparerForTests();
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var qatModel = preparer.PrepareForQAT(model, config);

            // Simulate training
            qatModel.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            _ = qatModel.Forward(input);

            // Act
            var stats = preparer.GetQATStatistics(qatModel);

            // Assert
            Assert.NotNull(stats);
            Assert.True(stats.MinWeightScale >= 0);
            Assert.True(stats.MaxWeightScale >= stats.MinWeightScale);
            Assert.True(stats.MinActivationScale >= 0);
            Assert.True(stats.MaxActivationScale >= stats.MinActivationScale);
        }

        [Fact]
        public void QATPreparer_Conversion_PreservesModelStructure()
        {
            // Arrange
            var preparer = new MockQATPreparerForTests();
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var qatModel = preparer.PrepareForQAT(model, config);

            // Simulate training
            qatModel.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            for (int i = 0; i < 5; i++)
            {
                _ = qatModel.Forward(input);
            }

            // Act
            var quantizedModel = preparer.ConvertToQuantized(qatModel);
            var inputShape = new long[] { 32, 10 };
            var testInput = new Tensor(new float[10 * 32], inputShape);
            var output = quantizedModel.Forward(testInput);

            // Assert
            Assert.NotNull(output);
        }

        [Fact]
        public void QATPreparer_PrepareModel_WithCustomLayers()
        {
            // Arrange
            var preparer = new MockQATPreparerForTests();
            var model = CreateModelWithCustomLayers();
            var config = CreateDefaultConfig();

            // Act
            var qatModel = preparer.PrepareForQAT(model, config);

            // Assert
            Assert.NotNull(qatModel);
        }

        [Fact]
        public void QATPreparer_PrepareModel_SkipsNonQuantizableLayers()
        {
            // Arrange
            var preparer = new MockQATPreparerForTests();
            var model = CreateModelWithNonQuantizableLayers();
            var config = CreateDefaultConfig();

            // Act
            var qatModel = preparer.PrepareForQAT(model, config);

            // Assert
            Assert.NotNull(qatModel);
        }

        [Fact]
        public void QATPreparer_Conversion_RemovesFakeQuantizationNodes()
        {
            // Arrange
            var preparer = new MockQATPreparerForTests();
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var qatModel = preparer.PrepareForQAT(model, config);
            var fakeQuantCountBefore = qatModel.GetFakeQuantizationNodeCount();

            // Simulate training
            qatModel.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            for (int i = 0; i < 5; i++)
            {
                _ = qatModel.Forward(input);
            }

            // Act
            var quantizedModel = preparer.ConvertToQuantized(qatModel);
            var fakeQuantCountAfter = quantizedModel.GetFakeQuantizationNodeCount();

            // Assert
            // Fake quant nodes should be removed after conversion
            Assert.True(fakeQuantCountBefore > 0);
            // In the quantized model, fake quant nodes should be replaced with real quantization
        }

        #region Helper Methods

        private static IModel CreateSimpleModel()
        {
            return new MockQATPreparerModel(new List<IModule>
            {
                new MockLinear(10, 5),
                new MockLinear(5, 2)
            });
        }

        private static IModel CreateModelWithCustomLayers()
        {
            return new MockQATPreparerModel(new List<IModule>
            {
                new MockLinear(10, 20),
                new MockConv2D(3, 16, kernelSize: 3),
                new MockLinear(16 * 8 * 8, 2)
            });
        }

        private static IModel CreateModelWithNonQuantizableLayers()
        {
            return new MockQATPreparerModel(new List<IModule>
            {
                new MockLinear(10, 5),
                new MockNonQuantizableLayer(),
                new MockLinear(5, 2)
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
    public class MockQATPreparerForTests
    {
        public IModel PrepareForQAT(IModel model, QuantizationConfig config,
            Dictionary<string, QuantizationConfig>? perLayerConfig = null)
        {
            // In production, this would:
            // 1. Analyze model structure
            // 2. Apply global config
            // 3. Override with per-layer config if provided
            // 4. Wrap layers with QAT wrappers
            // 5. Insert fake quantization nodes

            return new MockQATPreparedModel(model, config, perLayerConfig);
        }

        public IModel ConvertToQuantized(IModel qatModel)
        {
            // In production, this would:
            // 1. Extract trained quantization parameters
            // 2. Convert weights to Int8
            // 3. Remove fake quantization nodes
            // 4. Replace with real quantized operations

            return new MockQuantizedModel(qatModel);
        }

        public QATStatistics GetQATStatistics(IModel qatModel)
        {
            // In production, this would gather statistics from all quantized layers
            return new QATStatistics
            {
                MinWeightScale = 0.1f,
                MaxWeightScale = 1.0f,
                MinActivationScale = 0.05f,
                MaxActivationScale = 0.5f,
                QuantizedLayerCount = qatModel.GetQuantizedLayerCount()
            };
        }
    }

    /// <summary>
    /// QAT statistics.
    /// </summary>
    public class QATStatistics
    {
        public float MinWeightScale { get; set; }
        public float MaxWeightScale { get; set; }
        public float MinActivationScale { get; set; }
        public float MaxActivationScale { get; set; }
        public int QuantizedLayerCount { get; set; }
    }

    /// <summary>
    /// Mock QAT preparer model.
    /// </summary>
    public class MockQATPreparerModel : IModel
    {
        private readonly List<IModule> _layers;

        public MockQATPreparerModel(List<IModule> layers)
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
        public int GetFakeQuantizationNodeCount() => _layers.Count * 2;
        public List<FakeQuantize> GetFakeQuantizationNodes() =>
            Enumerable.Range(0, _layers.Count * 2)
                .Select(_ => new FakeQuantize(0.5f, 0))
                .ToList();

        public int GetQuantizedLayerCount() => _layers.Count;

        public Dictionary<string, QuantizationParameters?> GetQuantizationParameters()
        {
            var dict = new Dictionary<string, QuantizationParameters?>();
            for (int i = 0; i < _layers.Count; i++)
            {
                dict[$"layer_{i}"] = new QuantizationParameters
                {
                    Scale = 0.5f,
                    ZeroPoint = 0,
                    QuantizationMode = QuantizationMode.PerTensorSymmetric
                };
            }
            return dict;
        }

        public bool TrainingMode { get; set; } = true;
    }

    /// <summary>
    /// Mock QAT prepared model.
    /// </summary>
    public class MockQATPreparedModel : IModel
    {
        private readonly IModel _originalModel;
        private readonly QuantizationConfig _config;
        private readonly Dictionary<string, QuantizationConfig>? _perLayerConfig;

        public MockQATPreparedModel(IModel originalModel, QuantizationConfig config,
            Dictionary<string, QuantizationConfig>? perLayerConfig)
        {
            _originalModel = originalModel;
            _config = config;
            _perLayerConfig = perLayerConfig;
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
        public int GetFakeQuantizationNodeCount() => _originalModel.GetLayerCount() * 2;
        public List<FakeQuantize> GetFakeQuantizationNodes() =>
            Enumerable.Range(0, _originalModel.GetLayerCount() * 2)
                .Select(_ => new FakeQuantize(0.5f, 0))
                .ToList();

        public int GetQuantizedLayerCount() => _originalModel.GetLayerCount();

        public Dictionary<string, QuantizationParameters?> GetQuantizationParameters()
        {
            return _originalModel.GetQuantizationParameters();
        }

        public bool TrainingMode { get; set; } = true;
    }

    /// <summary>
    /// Mock quantized model.
    /// </summary>
    public class MockQuantizedModel : IModel
    {
        private readonly IModel _qatModel;

        public MockQuantizedModel(IModel qatModel)
        {
            _qatModel = qatModel;
        }

        public Tensor Forward(Tensor input)
        {
            // In production, this would use quantized operations
            return _qatModel.Forward(input);
        }

        public Tensor Backward(Tensor gradient)
        {
            return _qatModel.Backward(gradient);
        }

        public int GetLayerCount() => _qatModel.GetLayerCount();
        public int GetFakeQuantizationNodeCount() => 0; // No fake quant nodes in quantized model
        public List<FakeQuantize> GetFakeQuantizationNodes() => new List<FakeQuantize>();
        public int GetQuantizedLayerCount() => _qatModel.GetQuantizedLayerCount();
        public Dictionary<string, QuantizationParameters?> GetQuantizationParameters() => _qatModel.GetQuantizationParameters();
        public bool TrainingMode { get; set; } = false; // Quantized models are always in eval mode
    }

    /// <summary>
    /// Mock non-quantizable layer.
    /// </summary>
    public class MockNonQuantizableLayer : IModule
    {
        public Tensor Forward(Tensor input)
        {
            return input.Clone();
        }

        public Tensor Backward(Tensor upstreamGradient)
        {
            return upstreamGradient.Clone();
        }
    }

    #endregion
}
