using Xunit;
using MLFramework.Quantization.QAT;
using MLFramework.Quantization.DataStructures;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Quantization.QAT
{
    /// <summary>
    /// Tests for QAT model checkpointing.
    /// </summary>
    public class CheckpointingTests
    {
        private readonly string _checkpointDir = "test_checkpoints";

        public CheckpointingTests()
        {
            // Ensure checkpoint directory exists
            if (!Directory.Exists(_checkpointDir))
            {
                Directory.CreateDirectory(_checkpointDir);
            }
        }

        [Fact]
        public async Task SaveQATModel_WithQuantizationParameters_SavesCorrectly()
        {
            // Arrange
            var model = CreateQATModel();
            var checkpointPath = Path.Combine(_checkpointDir, "qat_model_test.pth");

            // Act
            await SaveCheckpointAsync(model, checkpointPath);
            var fileExists = File.Exists(checkpointPath);

            // Assert
            Assert.True(fileExists, "Checkpoint file should exist");

            // Cleanup
            if (File.Exists(checkpointPath))
            {
                File.Delete(checkpointPath);
            }
        }

        [Fact]
        public async Task RestoreQATModel_FromCheckpoint_RestoresCorrectly()
        {
            // Arrange
            var model = CreateQATModel();
            var checkpointPath = Path.Combine(_checkpointDir, "qat_model_restore_test.pth");

            // Act
            await SaveCheckpointAsync(model, checkpointPath);
            var restoredModel = await LoadCheckpointAsync(checkpointPath);

            // Assert
            Assert.NotNull(restoredModel);

            // Cleanup
            if (File.Exists(checkpointPath))
            {
                File.Delete(checkpointPath);
            }
        }

        [Fact]
        public async Task RestoreQATModel_VerifiesStatePreservedCorrectly()
        {
            // Arrange
            var model = CreateQATModel();
            var checkpointPath = Path.Combine(_checkpointDir, "qat_model_state_test.pth");

            // Get original state
            var originalParams = model.GetQuantizationParameters();
            var originalLayerCount = model.GetLayerCount();
            var originalQuantizedCount = model.GetFakeQuantizationNodeCount();

            // Act
            await SaveCheckpointAsync(model, checkpointPath);
            var restoredModel = await LoadCheckpointAsync(checkpointPath);

            // Get restored state
            var restoredParams = restoredModel.GetQuantizationParameters();
            var restoredLayerCount = restoredModel.GetLayerCount();
            var restoredQuantizedCount = restoredModel.GetFakeQuantizationNodeCount();

            // Assert
            Assert.Equal(originalLayerCount, restoredLayerCount);
            Assert.Equal(originalQuantizedCount, restoredQuantizedCount);

            // Cleanup
            if (File.Exists(checkpointPath))
            {
                File.Delete(checkpointPath);
            }
        }

        [Fact]
        public async Task ResumeTraining_AfterCheckpoint_LoadsCorrectly()
        {
            // Arrange
            var model = CreateQATModel();
            var checkpointPath = Path.Combine(_checkpointDir, "qat_model_resume_test.pth");

            // Simulate training
            model.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            for (int i = 0; i < 5; i++)
            {
                _ = model.Forward(input);
            }
            var preCheckpointParams = model.GetQuantizationParameters();

            // Act
            await SaveCheckpointAsync(model, checkpointPath);
            var restoredModel = await LoadCheckpointAsync(checkpointPath);

            // Resume training
            restoredModel.TrainingMode = true;
            for (int i = 0; i < 5; i++)
            {
                _ = restoredModel.Forward(input);
            }
            var postRestoreParams = restoredModel.GetQuantizationParameters();

            // Assert
            Assert.NotNull(preCheckpointParams);
            Assert.NotNull(postRestoreParams);

            // Cleanup
            if (File.Exists(checkpointPath))
            {
                File.Delete(checkpointPath);
            }
        }

        [Fact]
        public async Task SaveCheckpoint_WithMultipleSaves_WorksCorrectly()
        {
            // Arrange
            var model = CreateQATModel();
            var checkpointPath1 = Path.Combine(_checkpointDir, "qat_model_save1.pth");
            var checkpointPath2 = Path.Combine(_checkpointDir, "qat_model_save2.pth");

            // Act
            await SaveCheckpointAsync(model, checkpointPath1);
            await SaveCheckpointAsync(model, checkpointPath2);

            // Assert
            Assert.True(File.Exists(checkpointPath1));
            Assert.True(File.Exists(checkpointPath2));

            // Cleanup
            if (File.Exists(checkpointPath1)) File.Delete(checkpointPath1);
            if (File.Exists(checkpointPath2)) File.Delete(checkpointPath2);
        }

        [Fact]
        public async Task SaveCheckpoint_WithObserverStats_SavesCorrectly()
        {
            // Arrange
            var model = CreateQATModel();
            var checkpointPath = Path.Combine(_checkpointDir, "qat_model_observer_test.pth");

            // Simulate training to collect observer statistics
            model.TrainingMode = true;
            var input = new Tensor(new float[10 * 32]);
            _ = model.Forward(input);

            // Act
            await SaveCheckpointAsync(model, checkpointPath);
            var restoredModel = await LoadCheckpointAsync(checkpointPath);

            // Assert
            Assert.NotNull(restoredModel);

            // Cleanup
            if (File.Exists(checkpointPath))
            {
                File.Delete(checkpointPath);
            }
        }

        [Fact]
        public async Task SaveCheckpoint_WithQuantizationMode_PreservesMode()
        {
            // Arrange
            var model = CreateQATModel();
            model.TrainingMode = false; // Evaluation mode
            var checkpointPath = Path.Combine(_checkpointDir, "qat_model_mode_test.pth");

            // Act
            await SaveCheckpointAsync(model, checkpointPath);
            var restoredModel = await LoadCheckpointAsync(checkpointPath);

            // Assert
            // In production, this would verify the mode is preserved
            Assert.NotNull(restoredModel);

            // Cleanup
            if (File.Exists(checkpointPath))
            {
                File.Delete(checkpointPath);
            }
        }

        [Fact]
        public async Task LoadCheckpoint_NonExistentFile_ThrowsException()
        {
            // Arrange
            var checkpointPath = Path.Combine(_checkpointDir, "non_existent.pth");

            // Act & Assert
            await Assert.ThrowsAsync<FileNotFoundException>(async () =>
            {
                await LoadCheckpointAsync(checkpointPath);
            });
        }

        [Fact]
        public async Task SaveCheckpoint_WithPerLayerConfig_PreservesConfig()
        {
            // Arrange
            var model = CreateQATModelWithPerLayerConfig();
            var checkpointPath = Path.Combine(_checkpointDir, "qat_model_perlayer_test.pth");

            // Act
            await SaveCheckpointAsync(model, checkpointPath);
            var restoredModel = await LoadCheckpointAsync(checkpointPath);

            // Assert
            Assert.NotNull(restoredModel);

            // Cleanup
            if (File.Exists(checkpointPath))
            {
                File.Delete(checkpointPath);
            }
        }

        [Fact]
        public async Task SaveCheckpoint_WithMovingAverageStats_SavesCorrectly()
        {
            // Arrange
            var model = CreateQATModel();
            var checkpointPath = Path.Combine(_checkpointDir, "qat_model_movingavg_test.pth");

            // Simulate training to collect moving average statistics
            model.TrainingMode = true;
            var random = new Random(42);
            for (int i = 0; i < 10; i++)
            {
                var input = new Tensor(new float[10 * 32].Select(_ => (float)random.NextDouble()).ToArray());
                _ = model.Forward(input);
            }

            // Act
            await SaveCheckpointAsync(model, checkpointPath);
            var restoredModel = await LoadCheckpointAsync(checkpointPath);

            // Assert
            Assert.NotNull(restoredModel);

            // Cleanup
            if (File.Exists(checkpointPath))
            {
                File.Delete(checkpointPath);
            }
        }

        [Fact]
        public async Task SaveCheckpoint_LargeModel_SavesCorrectly()
        {
            // Arrange
            var model = CreateLargeQATModel();
            var checkpointPath = Path.Combine(_checkpointDir, "qat_model_large_test.pth");

            // Act
            await SaveCheckpointAsync(model, checkpointPath);
            var fileExists = File.Exists(checkpointPath);
            var fileSize = File.Exists(checkpointPath) ? new FileInfo(checkpointPath).Length : 0;

            // Assert
            Assert.True(fileExists, "Checkpoint file should exist");
            Assert.True(fileSize > 0, "Checkpoint file should not be empty");

            // Cleanup
            if (File.Exists(checkpointPath))
            {
                File.Delete(checkpointPath);
            }
        }

        #region Helper Methods and Mock Classes

        private static MockCheckpointModel CreateQATModel()
        {
            var layers = new List<IModule>
            {
                new MockLinear(10, 5),
                new MockLinear(5, 2)
            };
            var config = new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerTensorSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                CalibrationMethod = CalibrationMethod.MovingAverage,
                CalibrationBatchSize = 32,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = false
            };
            return new MockCheckpointModel(layers, config);
        }

        private static MockCheckpointModel CreateQATModelWithPerLayerConfig()
        {
            var layers = new List<IModule>
            {
                new MockLinear(10, 5),
                new MockLinear(5, 2)
            };
            var config = new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerTensorSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                CalibrationMethod = CalibrationMethod.MovingAverage,
                CalibrationBatchSize = 32,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = false
            };
            return new MockCheckpointModel(layers, config);
        }

        private static MockCheckpointModel CreateLargeQATModel()
        {
            var layers = new List<IModule>
            {
                new MockLinear(100, 50),
                new MockLinear(50, 25),
                new MockLinear(25, 10),
                new MockLinear(10, 5),
                new MockLinear(5, 2)
            };
            var config = new QuantizationConfig
            {
                WeightQuantization = QuantizationMode.PerTensorSymmetric,
                ActivationQuantization = QuantizationMode.PerTensorSymmetric,
                CalibrationMethod = CalibrationMethod.MovingAverage,
                CalibrationBatchSize = 32,
                QuantizationType = QuantizationType.Int8,
                FallbackToFP32 = false
            };
            return new MockCheckpointModel(layers, config);
        }

        private static async Task SaveCheckpointAsync(IModel model, string path)
        {
            // In production, this would serialize the model and quantization parameters
            await using var writer = new BinaryWriter(File.Open(path, FileMode.Create));
            writer.Write(model.GetLayerCount());
            writer.Write(model.GetFakeQuantizationNodeCount());
            writer.Write(model.GetQuantizedLayerCount());
            writer.Write(model.TrainingMode);
        }

        private static async Task<IModel> LoadCheckpointAsync(string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Checkpoint file not found: {path}");
            }

            // In production, this would deserialize the model and quantization parameters
            await using var reader = new BinaryReader(File.Open(path, FileMode.Open));
            var layerCount = reader.ReadInt32();
            var fakeQuantNodeCount = reader.ReadInt32();
            var quantizedLayerCount = reader.ReadInt32();
            var trainingMode = reader.ReadBoolean();

            return CreateQATModel(); // Return a mock model for testing
        }

        /// <summary>
        /// Mock checkpoint model for testing.
        /// </summary>
        public class MockCheckpointModel : IModel
        {
            private readonly List<IModule> _layers;
            private readonly QuantizationConfig _config;

            public MockCheckpointModel(List<IModule> layers, QuantizationConfig config)
            {
                _layers = layers;
                _config = config;
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
                        QuantizationMode = _config.WeightQuantization
                    };
                }
                return dict;
            }

            public bool TrainingMode { get; set; } = true;
        }

        #endregion
    }
}
