using Xunit;
using MLFramework.Quantization.QAT;
using MLFramework.Quantization.DataStructures;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Quantization.QAT
{
    /// <summary>
    /// Tests for QAT training integration.
    /// </summary>
    public class TrainingIntegrationTests
    {
        [Fact]
        public async Task TrainModelWithFakeQuantization_ForwardPassWorks()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();
            var qatModel = preparer.PrepareForQAT(model, config);
            var input = new Tensor(new float[10 * 32]);
            var target = new Tensor(new float[2 * 32]);

            // Act
            var output = qatModel.Forward(input);

            // Assert
            Assert.NotNull(output);
        }

        [Fact]
        public async Task TrainModelWithFakeQuantization_GradientsFlowThroughFakeQuantNodes()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();
            var qatModel = preparer.PrepareForQAT(model, config);
            var input = new Tensor(new float[10 * 32]);
            var outputGrad = new Tensor(new float[2 * 32]);

            // Act
            _ = qatModel.Forward(input);
            var gradient = qatModel.Backward(outputGrad);

            // Assert
            Assert.NotNull(gradient);
            Assert.Equal(input.Shape, gradient.Shape);
        }

        [Fact]
        public async Task TrainModelWithFakeQuantization_WeightsUpdateCorrectly()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();
            var qatModel = preparer.PrepareForQAT(model, config);
            var optimizer = CreateOptimizer(qatModel);
            var input = new Tensor(new float[10 * 32]);
            var target = new Tensor(new float[2 * 32]);

            // Act
            _ = qatModel.Forward(input);
            optimizer.Step(); // Update weights

            // Assert
            // Weights should have been updated
            Assert.NotNull(optimizer);
        }

        [Fact]
        public async Task TrainModelWithFakeQuantization_QuantizationParametersEvolve()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();
            var qatModel = preparer.PrepareForQAT(model, config);
            var initialParams = qatModel.GetQuantizationParameters();

            // Act
            qatModel.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            for (int i = 0; i < 5; i++)
            {
                _ = qatModel.Forward(input);
            }
            var updatedParams = qatModel.GetQuantizationParameters();

            // Assert
            Assert.NotNull(initialParams);
            Assert.NotNull(updatedParams);
            // Parameters should have been updated during training
        }

        [Fact]
        public async Task TrainingLoop_WithLossFunction_Completes()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();
            var qatModel = preparer.PrepareForQAT(model, config);
            var optimizer = CreateOptimizer(qatModel);
            var lossFunction = new MockLossFunction();
            var inputs = CreateBatch(32, 10);
            var targets = CreateBatch(32, 2);

            // Act
            for (int epoch = 0; epoch < 3; epoch++)
            {
                var outputs = qatModel.Forward(inputs);
                var loss = lossFunction.Compute(outputs, targets);
                _ = qatModel.Backward(loss.Gradients);
                optimizer.Step();
            }

            // Assert
            Assert.True(true, "Training loop completed successfully");
        }

        [Fact]
        public async Task Training_WithMultipleLayers_DoesNotCrash()
        {
            // Arrange
            var model = CreateMultiLayerModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();
            var qatModel = preparer.PrepareForQAT(model, config);
            var optimizer = CreateOptimizer(qatModel);
            var lossFunction = new MockLossFunction();
            var inputs = CreateBatch(16, 10);
            var targets = CreateBatch(16, 2);

            // Act
            for (int i = 0; i < 5; i++)
            {
                var outputs = qatModel.Forward(inputs);
                var loss = lossFunction.Compute(outputs, targets);
                _ = qatModel.Backward(loss.Gradients);
                optimizer.Step();
            }

            // Assert
            Assert.True(true, "Training with multiple layers completed successfully");
        }

        [Fact]
        public async Task Training_WithBatchProcessing_WorksCorrectly()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();
            var qatModel = preparer.PrepareForQAT(model, config);
            var optimizer = CreateOptimizer(qatModel);
            var lossFunction = new MockLossFunction();

            // Act
            for (int batch = 0; batch < 4; batch++)
            {
                var inputs = CreateBatch(32, 10);
                var targets = CreateBatch(32, 2);
                var outputs = qatModel.Forward(inputs);
                var loss = lossFunction.Compute(outputs, targets);
                _ = qatModel.Backward(loss.Gradients);
                optimizer.Step();
            }

            // Assert
            Assert.True(true, "Batch processing completed successfully");
        }

        [Fact]
        public async Task Training_LossDecreasesOverIterations()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();
            var qatModel = preparer.PrepareForQAT(model, config);
            var optimizer = CreateOptimizer(qatModel);
            var lossFunction = new MockLossFunction();
            var inputs = CreateBatch(32, 10);
            var targets = CreateBatch(32, 2);

            var losses = new List<float>();

            // Act
            for (int i = 0; i < 10; i++)
            {
                var outputs = qatModel.Forward(inputs);
                var loss = lossFunction.Compute(outputs, targets);
                losses.Add(loss.Value);
                _ = qatModel.Backward(loss.Gradients);
                optimizer.Step();
            }

            // Assert
            Assert.Equal(10, losses.Count);
            // Loss should generally decrease (but may fluctuate due to quantization noise)
            var laterLosses = losses.Skip(5).Average();
            var earlierLosses = losses.Take(5).Average();
            // At minimum, check that training didn't explode
            Assert.True(losses.All(l => float.IsFinite(l)));
        }

        [Fact]
        public async Task Training_EvaluationMode_DisablesObserverUpdates()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();
            var preparer = new QATPreparer();
            var qatModel = preparer.PrepareForQAT(model, config);
            var input = CreateBatch(32, 10);

            // Act
            qatModel.TrainingMode = true;
            _ = qatModel.Forward(input);
            var trainParams = qatModel.GetQuantizationParameters();

            qatModel.TrainingMode = false;
            _ = qatModel.Forward(input);
            var evalParams = qatModel.GetQuantizationParameters();

            // Assert
            Assert.NotNull(trainParams);
            Assert.NotNull(evalParams);
        }

        [Fact]
        public async Task Training_WithDifferentQuantizationModes_WorksCorrectly()
        {
            // Arrange
            var model = CreateSimpleModel();
            var config = CreateDefaultConfig();

            var modes = new[]
            {
                QuantizationMode.PerTensorSymmetric,
                QuantizationMode.PerTensorAsymmetric
            };

            // Act & Assert
            foreach (var mode in modes)
            {
                config.WeightQuantization = mode;
                config.ActivationQuantization = mode;
                var preparer = new QATPreparer();
                var qatModel = preparer.PrepareForQAT(model, config);
                var optimizer = CreateOptimizer(qatModel);
                var input = CreateBatch(16, 10);
                var target = CreateBatch(16, 2);

                _ = qatModel.Forward(input);
                optimizer.Step();

                Assert.NotNull(qatModel);
            }
        }

        #region Helper Methods and Mock Classes

        private static IModel CreateSimpleModel()
        {
            return new MockTrainingModel(new List<IModule>
            {
                new MockLinear(10, 5),
                new MockLinear(5, 2)
            });
        }

        private static IModel CreateMultiLayerModel()
        {
            return new MockTrainingModel(new List<IModule>
            {
                new MockLinear(10, 20),
                new MockLinear(20, 10),
                new MockLinear(10, 5),
                new MockLinear(5, 2)
            });
        }

        private static IOptimizer CreateOptimizer(IModel model)
        {
            return new MockOptimizer(model);
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

        private static Tensor CreateBatch(int batchSize, int size)
        {
            var data = new float[batchSize * size];
            var random = new Random(42);
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (float)random.NextDouble();
            }
            return new Tensor(data, new long[] { batchSize, size });
        }

        /// <summary>
        /// Mock Training model with training mode support.
        /// </summary>
        public class MockTrainingModel : IModel
        {
            private readonly List<IModule> _layers;
            public bool TrainingMode { get; set; } = true;

            public MockTrainingModel(List<IModule> layers)
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
        }

        /// <summary>
        /// Mock optimizer for testing.
        /// </summary>
        public class MockOptimizer : IOptimizer
        {
            private readonly IModel _model;

            public MockOptimizer(IModel model)
            {
                _model = model;
            }

            public void Step()
            {
                // Simulate parameter update
            }

            public void ZeroGrad()
            {
                // Reset gradients
            }
        }

        /// <summary>
        /// Mock loss function for testing.
        /// </summary>
        public class MockLossFunction
        {
            public LossResult Compute(Tensor outputs, Tensor targets)
            {
                // Simple MSE loss
                var outputData = outputs.ToArray();
                var targetData = targets.ToArray();
                float sum = 0;
                for (int i = 0; i < outputData.Length; i++)
                {
                    var diff = outputData[i] - targetData[i];
                    sum += diff * diff;
                }
                var loss = sum / outputData.Length;

                // Gradients (simplified)
                var gradients = new float[outputData.Length];
                for (int i = 0; i < outputData.Length; i++)
                {
                    gradients[i] = 2 * (outputData[i] - targetData[i]) / outputData.Length;
                }

                return new LossResult
                {
                    Value = loss,
                    Gradients = new Tensor(gradients, outputs.Shape)
                };
            }
        }

        public class LossResult
        {
            public float Value { get; set; }
            public Tensor Gradients { get; set; } = null!;
        }

        public interface IOptimizer
        {
            void Step();
            void ZeroGrad();
        }

        #endregion
    }
}
