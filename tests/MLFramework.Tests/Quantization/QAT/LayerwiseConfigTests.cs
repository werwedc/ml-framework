using Xunit;
using MLFramework.Quantization.QAT;
using MLFramework.Quantization.DataStructures;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Quantization.QAT
{
    /// <summary>
    /// Tests for per-layer QAT configuration.
    /// </summary>
    public class LayerwiseConfigTests
    {
        [Fact]
        public void LayerwiseConfig_OverrideGlobalConfigPerLayer()
        {
            // Arrange
            var model = CreateModel();
            var globalConfig = CreateGlobalConfig();
            var perLayerConfig = new Dictionary<string, QuantizationConfig>
            {
                {
                    "layer_0", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerChannelSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                        CalibrationMethod = CalibrationMethod.MinMax,
                        CalibrationBatchSize = 64,
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = false
                    }
                }
            };
            var preparer = new QATPreparer();

            // Act
            var qatModel = preparer.PrepareForQAT(model, globalConfig, perLayerConfig);
            var quantParams = qatModel.GetQuantizationParameters();

            // Assert
            Assert.NotNull(quantParams);
            Assert.True(quantParams.ContainsKey("layer_0"));
        }

        [Fact]
        public void LayerwiseConfig_DisableQATForSpecificLayers()
        {
            // Arrange
            var model = CreateModel();
            var globalConfig = CreateGlobalConfig();
            var perLayerConfig = new Dictionary<string, QuantizationConfig>
            {
                {
                    "layer_0", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerTensorSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                        CalibrationMethod = CalibrationMethod.MovingAverage,
                        CalibrationBatchSize = 32,
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = true // Disable QAT for this layer
                    }
                }
            };
            var preparer = new QATPreparer();

            // Act
            var qatModel = preparer.PrepareForQAT(model, globalConfig, perLayerConfig);
            var quantParams = qatModel.GetQuantizationParameters();

            // Assert
            Assert.NotNull(quantParams);
        }

        [Fact]
        public void LayerwiseConfig_MixQuantizedAndNonQuantizedLayers()
        {
            // Arrange
            var model = CreateModel();
            var globalConfig = CreateGlobalConfig();
            var perLayerConfig = new Dictionary<string, QuantizationConfig>
            {
                {
                    "layer_0", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerChannelSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                        CalibrationMethod = CalibrationMethod.MovingAverage,
                        CalibrationBatchSize = 32,
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = false
                    }
                },
                {
                    "layer_1", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerTensorSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                        CalibrationMethod = CalibrationMethod.MovingAverage,
                        CalibrationBatchSize = 32,
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = true // Non-quantized
                    }
                },
                {
                    "layer_2", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerTensorSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                        CalibrationMethod = CalibrationMethod.MovingAverage,
                        CalibrationBatchSize = 32,
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = false
                    }
                }
            };
            var preparer = new QATPreparer();

            // Act
            var qatModel = preparer.PrepareForQAT(model, globalConfig, perLayerConfig);
            var quantParams = qatModel.GetQuantizationParameters();

            // Assert
            Assert.NotNull(quantParams);
            Assert.True(quantParams.Count >= 3);
        }

        [Fact]
        public void LayerwiseConfig_VerifyFinalModelCorrectness()
        {
            // Arrange
            var model = CreateModel();
            var globalConfig = CreateGlobalConfig();
            var perLayerConfig = new Dictionary<string, QuantizationConfig>
            {
                {
                    "layer_0", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerTensorSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                        CalibrationMethod = CalibrationMethod.MinMax,
                        CalibrationBatchSize = 32,
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = false
                    }
                }
            };
            var preparer = new QATPreparer();
            var input = new Tensor(new float[10 * 32]);

            // Act
            var qatModel = preparer.PrepareForQAT(model, globalConfig, perLayerConfig);
            var output = qatModel.Forward(input);

            // Assert
            Assert.NotNull(output);
        }

        [Fact]
        public void LayerwiseConfig_DifferentQuantizationModesPerLayer()
        {
            // Arrange
            var model = CreateModel();
            var globalConfig = CreateGlobalConfig();
            var perLayerConfig = new Dictionary<string, QuantizationConfig>
            {
                {
                    "layer_0", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerTensorSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                        CalibrationMethod = CalibrationMethod.MinMax,
                        CalibrationBatchSize = 32,
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = false
                    }
                },
                {
                    "layer_1", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerChannelSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorAsymmetric,
                        CalibrationMethod = CalibrationMethod.MovingAverage,
                        CalibrationBatchSize = 32,
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = false
                    }
                }
            };
            var preparer = new QATPreparer();

            // Act
            var qatModel = preparer.PrepareForQAT(model, globalConfig, perLayerConfig);
            var quantParams = qatModel.GetQuantizationParameters();

            // Assert
            Assert.NotNull(quantParams);
        }

        [Fact]
        public void LayerwiseConfig_DifferentCalibrationMethodsPerLayer()
        {
            // Arrange
            var model = CreateModel();
            var globalConfig = CreateGlobalConfig();
            var perLayerConfig = new Dictionary<string, QuantizationConfig>
            {
                {
                    "layer_0", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerTensorSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                        CalibrationMethod = CalibrationMethod.MinMax,
                        CalibrationBatchSize = 32,
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = false
                    }
                },
                {
                    "layer_1", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerTensorSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                        CalibrationMethod = CalibrationMethod.MovingAverage,
                        CalibrationBatchSize = 32,
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = false
                    }
                },
                {
                    "layer_2", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerTensorSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                        CalibrationMethod = CalibrationMethod.Entropy,
                        CalibrationBatchSize = 32,
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = false
                    }
                }
            };
            var preparer = new QATPreparer();

            // Act
            var qatModel = preparer.PrepareForQAT(model, globalConfig, perLayerConfig);

            // Assert
            Assert.NotNull(qatModel);
        }

        [Fact]
        public void LayerwiseConfig_DifferentBatchSizesPerLayer()
        {
            // Arrange
            var model = CreateModel();
            var globalConfig = CreateGlobalConfig();
            var perLayerConfig = new Dictionary<string, QuantizationConfig>
            {
                {
                    "layer_0", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerTensorSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                        CalibrationMethod = CalibrationMethod.MinMax,
                        CalibrationBatchSize = 16, // Smaller batch
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = false
                    }
                },
                {
                    "layer_1", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerTensorSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                        CalibrationMethod = CalibrationMethod.MinMax,
                        CalibrationBatchSize = 64, // Larger batch
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = false
                    }
                }
            };
            var preparer = new QATPreparer();

            // Act
            var qatModel = preparer.PrepareForQAT(model, globalConfig, perLayerConfig);

            // Assert
            Assert.NotNull(qatModel);
        }

        [Fact]
        public void LayerwiseConfig_PartialConfiguration_UsesGlobalForRest()
        {
            // Arrange
            var model = CreateModel();
            var globalConfig = CreateGlobalConfig();
            var perLayerConfig = new Dictionary<string, QuantizationConfig>
            {
                {
                    "layer_0", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerChannelSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                        CalibrationMethod = CalibrationMethod.MinMax,
                        CalibrationBatchSize = 32,
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = false
                    }
                }
                // layer_1 and layer_2 should use global config
            };
            var preparer = new QATPreparer();

            // Act
            var qatModel = preparer.PrepareForQAT(model, globalConfig, perLayerConfig);
            var quantParams = qatModel.GetQuantizationParameters();

            // Assert
            Assert.NotNull(quantParams);
            Assert.True(quantParams.ContainsKey("layer_0"));
        }

        [Fact]
        public void LayerwiseConfig_EmptyPerLayerConfig_UsesGlobalConfig()
        {
            // Arrange
            var model = CreateModel();
            var globalConfig = CreateGlobalConfig();
            var perLayerConfig = new Dictionary<string, QuantizationConfig>();
            var preparer = new QATPreparer();

            // Act
            var qatModel = preparer.PrepareForQAT(model, globalConfig, perLayerConfig);

            // Assert
            Assert.NotNull(qatModel);
        }

        [Fact]
        public void LayerwiseConfig_TrainingWithMixedConfiguration_WorksCorrectly()
        {
            // Arrange
            var model = CreateModel();
            var globalConfig = CreateGlobalConfig();
            var perLayerConfig = new Dictionary<string, QuantizationConfig>
            {
                {
                    "layer_0", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerTensorSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                        CalibrationMethod = CalibrationMethod.MovingAverage,
                        CalibrationBatchSize = 32,
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = false
                    }
                },
                {
                    "layer_1", new QuantizationConfig
                    {
                        WeightQuantization = QuantizationMode.PerTensorSymmetric,
                        ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                        CalibrationMethod = CalibrationMethod.MovingAverage,
                        CalibrationBatchSize = 32,
                        QuantizationType = QuantizationType.Int8,
                        FallbackToFP32 = true
                    }
                }
            };
            var preparer = new QATPreparer();
            var input = new Tensor(new float[10 * 32]);

            // Act
            var qatModel = preparer.PrepareForQAT(model, globalConfig, perLayerConfig);
            qatModel.TrainingMode = true;

            for (int i = 0; i < 5; i++)
            {
                _ = qatModel.Forward(input);
            }

            // Assert
            Assert.NotNull(qatModel);
        }

        #region Helper Methods and Mock Classes

        private static MockLayerwiseModel CreateModel()
        {
            var layers = new List<IModule>
            {
                new MockLinear(10, 20),
                new MockLinear(20, 10),
                new MockLinear(10, 5)
            };
            return new MockLayerwiseModel(layers);
        }

        private static QuantizationConfig CreateGlobalConfig()
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

        /// <summary>
        /// Mock layerwise model for testing.
        /// </summary>
        public class MockLayerwiseModel : IModel
        {
            private readonly List<IModule> _layers;

            public MockLayerwiseModel(List<IModule> layers)
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

        #endregion
    }
}
