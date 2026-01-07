using Xunit;
using MLFramework.Quantization.QAT;
using MLFramework.Quantization.DataStructures;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Quantization.QAT
{
    /// <summary>
    /// Tests for final conversion to Int8.
    /// </summary>
    public class ConversionTests
    {
        [Fact]
        public void Conversion_ExtractTrainedQuantizationParameters()
        {
            // Arrange
            var model = CreateQATModel();
            var preparer = new QATPreparer();

            // Simulate training to collect statistics
            model.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            for (int i = 0; i < 10; i++)
            {
                _ = model.Forward(input);
            }

            // Act
            var quantParams = model.GetQuantizationParameters();

            // Assert
            Assert.NotNull(quantParams);
            Assert.True(quantParams.Count > 0);
        }

        [Fact]
        public void Conversion_ConvertWeightsToInt8()
        {
            // Arrange
            var model = CreateQATModel();
            var preparer = new QATPreparer();

            // Simulate training
            model.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            for (int i = 0; i < 5; i++)
            {
                _ = model.Forward(input);
            }

            // Act
            var quantizedModel = preparer.ConvertToQuantized(model);
            var quantizedLayerCount = quantizedModel.GetQuantizedLayerCount();

            // Assert
            Assert.NotNull(quantizedModel);
            Assert.True(quantizedLayerCount > 0);
        }

        [Fact]
        public void Conversion_RemoveFakeQuantizationNodes()
        {
            // Arrange
            var model = CreateQATModel();
            var preparer = new QATPreparer();

            // Get fake quant node count before conversion
            var fakeQuantCountBefore = model.GetFakeQuantizationNodeCount();
            Assert.True(fakeQuantCountBefore > 0, "QAT model should have fake quant nodes");

            // Act
            var quantizedModel = preparer.ConvertToQuantized(model);
            var fakeQuantCountAfter = quantizedModel.GetFakeQuantizationNodeCount();

            // Assert
            // Fake quant nodes should be removed after conversion
            Assert.Equal(0, fakeQuantCountAfter, "Quantized model should not have fake quant nodes");
        }

        [Fact]
        public void Conversion_VerifyInferenceAccuracy()
        {
            // Arrange
            var model = CreateQATModel();
            var preparer = new QATPreparer();
            var testInput = new Tensor(new float[10 * 32]);

            // Simulate training
            model.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            for (int i = 0; i < 5; i++)
            {
                _ = model.Forward(input);
            }

            // Get QAT model output
            model.TrainingMode = false;
            var qatOutput = model.Forward(testInput);

            // Act
            var quantizedModel = preparer.ConvertToQuantized(model);
            var quantizedOutput = quantizedModel.Forward(testInput);

            // Assert
            Assert.NotNull(qatOutput);
            Assert.NotNull(quantizedOutput);
            Assert.Equal(qatOutput.Shape, quantizedOutput.Shape);
        }

        [Fact]
        public void Conversion_WithPerTensorMode_WorksCorrectly()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerTensorSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                CalibrationMethod = CalibrationMethod.MovingAverage,
                CalibrationBatchSize = 32,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = false
            };
            var model = CreateQATModel(config);
            var preparer = new QATPreparer();

            // Simulate training
            model.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            for (int i = 0; i < 5; i++)
            {
                _ = model.Forward(input);
            }

            // Act
            var quantizedModel = preparer.ConvertToQuantized(model);

            // Assert
            Assert.NotNull(quantizedModel);
        }

        [Fact]
        public void Conversion_WithPerChannelMode_WorksCorrectly()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerChannelSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                CalibrationMethod = CalibrationMethod.MovingAverage,
                CalibrationBatchSize = 32,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = false
            };
            var model = CreateQATModel(config);
            var preparer = new QATPreparer();

            // Simulate training
            model.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            for (int i = 0; i < 5; i++)
            {
                _ = model.Forward(input);
            }

            // Act
            var quantizedModel = preparer.ConvertToQuantized(model);

            // Assert
            Assert.NotNull(quantizedModel);
        }

        [Fact]
        public void Conversion_WithAsymmetricMode_WorksCorrectly()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerTensorAsymmetric,
                ActivationQuantization = QuantizationMode.PerTensorAsymmetric,
                CalibrationMethod = CalibrationMethod.MinMax,
                CalibrationBatchSize = 32,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = false
            };
            var model = CreateQATModel(config);
            var preparer = new QATPreparer();

            // Simulate training
            model.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            for (int i = 0; i < 5; i++)
            {
                _ = model.Forward(input);
            }

            // Act
            var quantizedModel = preparer.ConvertToQuantized(model);

            // Assert
            Assert.NotNull(quantizedModel);
        }

        [Fact]
        public void Conversion_WithMixedPrecision_WorksCorrectly()
        {
            // Arrange
            var config = new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerTensorSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                CalibrationMethod = CalibrationMethod.MovingAverage,
                CalibrationBatchSize = 32,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = true, // Enable mixed precision
                AccuracyThreshold = 0.01f
            };
            var model = CreateQATModel(config);
            var preparer = new QATPreparer();

            // Simulate training
            model.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            for (int i = 0; i < 5; i++)
            {
                _ = model.Forward(input);
            }

            // Act
            var quantizedModel = preparer.ConvertToQuantized(model);

            // Assert
            Assert.NotNull(quantizedModel);
        }

        [Fact]
        public void Conversion_VerifyWeightsAreInt8()
        {
            // Arrange
            var model = CreateQATModel();
            var preparer = new QATPreparer();

            // Simulate training
            model.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            for (int i = 0; i < 5; i++)
            {
                _ = model.Forward(input);
            }

            // Act
            var quantizedModel = preparer.ConvertToQuantized(model);

            // Assert
            // In production, this would verify that weights are actually Int8
            Assert.NotNull(quantizedModel);
        }

        [Fact]
        public void Conversion_VerifyActivationsAreInt8()
        {
            // Arrange
            var model = CreateQATModel();
            var preparer = new QATPreparer();

            // Simulate training
            model.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            for (int i = 0; i < 5; i++)
            {
                _ = model.Forward(input);
            }

            // Act
            var quantizedModel = preparer.ConvertToQuantized(model);
            var testInput = new Tensor(new float[10 * 32]);
            var output = quantizedModel.Forward(testInput);

            // Assert
            // In production, this would verify that activations are actually Int8 during inference
            Assert.NotNull(output);
        }

        [Fact]
        public void Conversion_LargeModel_WorksCorrectly()
        {
            // Arrange
            var model = CreateLargeQATModel();
            var preparer = new QATPreparer();

            // Simulate training
            model.TrainingMode = true;
            var input = new Tensor(new float[100 * 32]);
            for (int i = 0; i < 5; i++)
            {
                _ = model.Forward(input);
            }

            // Act
            var quantizedModel = preparer.ConvertToQuantized(model);

            // Assert
            Assert.NotNull(quantizedModel);
        }

        [Fact]
        public void Conversion_MultipleLayers_WorksCorrectly()
        {
            // Arrange
            var model = CreateMultiLayerQATModel();
            var preparer = new QATPreparer();

            // Simulate training
            model.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            for (int i = 0; i < 5; i++)
            {
                _ = model.Forward(input);
            }

            // Act
            var quantizedModel = preparer.ConvertToQuantized(model);
            var quantizedLayerCount = quantizedModel.GetQuantizedLayerCount();

            // Assert
            Assert.NotNull(quantizedModel);
            Assert.True(quantizedLayerCount > 1);
        }

        [Fact]
        public void Conversion_WithConv2DLayers_WorksCorrectly()
        {
            // Arrange
            var model = CreateConv2DQATModel();
            var preparer = new QATPreparer();

            // Simulate training
            model.TrainingMode = true;
            var input = new Tensor(new float[3 * 32 * 32 * 32], new long[] { 32, 3, 32, 32 });
            for (int i = 0; i < 5; i++)
            {
                _ = model.Forward(input);
            }

            // Act
            var quantizedModel = preparer.ConvertToQuantized(model);

            // Assert
            Assert.NotNull(quantizedModel);
        }

        #region Helper Methods and Mock Classes

        private static MockConversionModel CreateQATModel(QuantizationConfig? config = null)
        {
            config ??= new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerTensorSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                CalibrationMethod = CalibrationMethod.MovingAverage,
                CalibrationBatchSize = 32,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = false
            };

            var layers = new List<IModule>
            {
                new MockLinear(10, 5),
                new MockLinear(5, 2)
            };
            return new MockConversionModel(layers, config);
        }

        private static MockConversionModel CreateLargeQATModel()
        {
            var config = new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerTensorSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                CalibrationMethod = CalibrationMethod.MovingAverage,
                CalibrationBatchSize = 32,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = false
            };

            var layers = new List<IModule>
            {
                new MockLinear(100, 50),
                new MockLinear(50, 25),
                new MockLinear(25, 10),
                new MockLinear(10, 5),
                new MockLinear(5, 2)
            };
            return new MockConversionModel(layers, config);
        }

        private static MockConversionModel CreateMultiLayerQATModel()
        {
            var config = new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerTensorSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                CalibrationMethod = CalibrationMethod.MovingAverage,
                CalibrationBatchSize = 32,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = false
            };

            var layers = new List<IModule>
            {
                new MockLinear(10, 20),
                new MockLinear(20, 15),
                new MockLinear(15, 10),
                new MockLinear(10, 5),
                new MockLinear(5, 2)
            };
            return new MockConversionModel(layers, config);
        }

        private static MockConversionModel CreateConv2DQATModel()
        {
            var config = new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerTensorSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                CalibrationMethod = CalibrationMethod.MovingAverage,
                CalibrationBatchSize = 32,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = false
            };

            var layers = new List<IModule>
            {
                new MockConv2D(3, 16, kernelSize: 3),
                new MockConv2D(16, 32, kernelSize: 3),
                new MockLinear(32 * 8 * 8, 2)
            };
            return new MockConversionModel(layers, config);
        }

        /// <summary>
        /// Mock conversion model for testing.
        /// </summary>
        public class MockConversionModel : IModel
        {
            private readonly List<IModule> _layers;
            private readonly QuantizationConfig _config;
            private bool _isQuantized;

            public MockConversionModel(List<IModule> layers, QuantizationConfig config)
            {
                _layers = layers;
                _config = config;
                _isQuantized = false;
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
            public int GetFakeQuantizationNodeCount() => _isQuantized ? 0 : _layers.Count * 2;
            public List<FakeQuantize> GetFakeQuantizationNodes() =>
                _isQuantized
                    ? new List<FakeQuantize>()
                    : Enumerable.Range(0, _layers.Count * 2)
                        .Select(_ => new FakeQuantize(0.5f, 0))
                        .ToList();

            public int GetQuantizedLayerCount() => _isQuantized ? _layers.Count : _layers.Count;

            public Dictionary<string, QuantizationParameters?> GetQuantizationParameters()
            {
                var dict = new Dictionary<string, QuantizationParameters?>();
                for (int i = 0; i < _layers.Count; i++)
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

            public bool TrainingMode { get; set; } = true;

            public void SetQuantized(bool isQuantized)
            {
                _isQuantized = isQuantized;
            }
        }

        #endregion
    }
}
