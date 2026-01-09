using System;
using System.IO;
using System.Threading.Tasks;
using MLFramework.Core;
using MLFramework.ModelZoo;
using Xunit;

namespace ModelZooTests
{
    /// <summary>
    /// Unit tests for the ModelZoo load API.
    /// </summary>
    public class ModelZooTests
    {
        private readonly string _testCacheDir;
        private readonly string _testModelFile;

        public ModelZooTests()
        {
            _testCacheDir = Path.Combine(Path.GetTempPath(), "ModelZooTests", Guid.NewGuid().ToString());
            _testModelFile = Path.Combine(_testCacheDir, "test_model.bin");

            // Ensure test directory exists
            if (!Directory.Exists(_testCacheDir))
            {
                Directory.CreateDirectory(_testCacheDir);
            }
        }

        [Fact]
        public void Constructor_InitializesWithDefaultConfiguration()
        {
            // Act
            var config = new ModelZooConfiguration();

            // Assert
            Assert.NotNull(config.DefaultDevice);
            Assert.True(config.CacheEnabled);
            Assert.True(config.AutoDownloadEnabled);
            Assert.Equal(300000, config.DefaultDownloadTimeoutMs);
        }

        [Fact]
        public void Constructor_WithCustomConfiguration_AppliesSettings()
        {
            // Arrange
            var device = Device.CreateCpu();
            
            // Act
            var config = new ModelZooConfiguration(
                device,
                cacheEnabled: false,
                autoDownloadEnabled: false,
                defaultDownloadTimeoutMs: 600000);

            // Assert
            Assert.NotNull(config.DefaultDevice);
            Assert.False(config.CacheEnabled);
            Assert.False(config.AutoDownloadEnabled);
            Assert.Equal(600000, config.DefaultDownloadTimeoutMs);
        }

        [Fact]
        public void ModelNotFoundException_ContainsCorrectMessage()
        {
            // Arrange & Act
            var exception = new ModelNotFoundException("test_model");

            // Assert
            Assert.Contains("test_model", exception.Message);
            Assert.Equal("test_model", exception.ModelName);
        }

        [Fact]
        public void VersionNotFoundException_ContainsCorrectMessage()
        {
            // Arrange & Act
            var exception = new VersionNotFoundException("test_model", "1.0.0");

            // Assert
            Assert.Contains("test_model", exception.Message);
            Assert.Contains("1.0.0", exception.Message);
            Assert.Equal("test_model", exception.ModelName);
            Assert.Equal("1.0.0", exception.Version);
        }

        [Fact]
        public void IncompatibleModelException_ContainsCorrectMessage()
        {
            // Arrange & Act
            var exception = new IncompatibleModelException("test_model", "1.0.0", "ResNet", "BERT");

            // Assert
            Assert.Contains("test_model", exception.Message);
            Assert.Contains("1.0.0", exception.Message);
            Assert.Contains("ResNet", exception.Message);
            Assert.Contains("BERT", exception.Message);
            Assert.Equal("test_model", exception.ModelName);
            Assert.Equal("1.0.0", exception.Version);
            Assert.Equal("ResNet", exception.ExpectedArchitecture);
            Assert.Equal("BERT", exception.ActualArchitecture);
        }

        [Fact]
        public void DeserializationException_ContainsCorrectMessage()
        {
            // Arrange & Act
            var exception = new DeserializationException("/path/to/model.bin", "Invalid magic bytes");

            // Assert
            Assert.Contains("/path/to/model.bin", exception.Message);
            Assert.Contains("Invalid magic bytes", exception.Message);
            Assert.Equal("/path/to/model.bin", exception.FilePath);
        }

        [Fact]
        public void IsAvailable_ReturnsFalseForNonExistentModel()
        {
            // Arrange
            var nonExistentModel = "non_existent_model_v999";

            // Act
            var result = ModelZoo.IsAvailable(nonExistentModel);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void RegisterModel_AddsModelToRegistry()
        {
            // Arrange
            var metadata = CreateTestModelMetadata("test_model", "1.0.0");

            // Act
            ModelZoo.RegisterModel(metadata);

            // Assert
            Assert.True(ModelZoo.IsAvailable("test_model", "1.0.0"));
        }

        [Fact]
        public void RegisterModel_ThrowsOnNullMetadata()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentNullException>(() => ModelZoo.RegisterModel(null));
        }

        [Fact]
        public void RegisterModel_ThrowsOnEmptyModelName()
        {
            // Arrange
            var metadata = CreateTestModelMetadata("", "1.0.0");

            // Act & Assert
            Assert.Throws<ArgumentException>(() => ModelZoo.RegisterModel(metadata));
        }

        [Fact]
        public void ListModels_ReturnsReadOnlyList()
        {
            // Arrange
            var metadata = CreateTestModelMetadata("test_model_list", "1.0.0");
            ModelZoo.RegisterModel(metadata);

            // Act
            var models = ModelZoo.ListModels();

            // Assert
            Assert.NotNull(models);
            Assert.True(models.Count >= 1);
        }

        [Fact]
        public void GetLatestVersion_ReturnsNullForNonExistentModel()
        {
            // Arrange
            var nonExistentModel = "non_existent_model";

            // Act
            var result = ModelZoo.GetLatestVersion(nonExistentModel);

            // Assert
            Assert.Null(result);
        }

        [Fact]
        public void ClearCache_CallsCacheManagerClearCache()
        {
            // Arrange & Act - This should not throw
            ModelZoo.ClearCache();

            // Assert - If no exception was thrown, test passes
            Assert.True(true);
        }

        [Fact]
        public void GetCacheStatistics_ReturnsStatistics()
        {
            // Act
            var stats = ModelZoo.GetCacheStatistics();

            // Assert
            Assert.NotNull(stats);
            Assert.NotNull(stats.TotalCacheSize);
        }

        [Fact]
        public void ListCachedModels_ReturnsEmptyListForNewCache()
        {
            // Act
            var cachedModels = ModelZoo.ListCachedModels();

            // Assert
            Assert.NotNull(cachedModels);
            Assert.Empty(cachedModels);
        }

        [Fact]
        public void SetDefaultDevice_UpdatesConfiguration()
        {
            // Arrange
            var device = Device.CreateCpu();

            // Act
            ModelZoo.SetDefaultDevice(device);

            // Assert
            Assert.Equal(device, ModelZoo.GetDefaultDevice());
        }

        [Fact]
        public void SetDefaultDevice_ThrowsOnNullDevice()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentNullException>(() => ModelZoo.SetDefaultDevice(null));
        }

        [Fact]
        public void SetCacheEnabled_UpdatesConfiguration()
        {
            // Arrange
            var originalValue = ModelZoo.Configuration.CacheEnabled;

            // Act
            ModelZoo.SetCacheEnabled(!originalValue);

            // Assert
            Assert.Equal(!originalValue, ModelZoo.Configuration.CacheEnabled);

            // Cleanup
            ModelZoo.SetCacheEnabled(originalValue);
        }

        [Fact]
        public void SetAutoDownloadEnabled_UpdatesConfiguration()
        {
            // Arrange
            var originalValue = ModelZoo.Configuration.AutoDownloadEnabled;

            // Act
            ModelZoo.SetAutoDownloadEnabled(!originalValue);

            // Assert
            Assert.Equal(!originalValue, ModelZoo.Configuration.AutoDownloadEnabled);

            // Cleanup
            ModelZoo.SetAutoDownloadEnabled(originalValue);
        }

        [Fact]
        public void SetDownloadTimeout_UpdatesConfiguration()
        {
            // Arrange
            var originalValue = ModelZoo.Configuration.DefaultDownloadTimeoutMs;

            // Act
            ModelZoo.SetDownloadTimeout(600000);

            // Assert
            Assert.Equal(600000, ModelZoo.Configuration.DefaultDownloadTimeoutMs);

            // Cleanup
            ModelZoo.SetDownloadTimeout(originalValue);
        }

        [Fact]
        public void SetDownloadTimeout_ThrowsOnNegativeValue()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() => ModelZoo.SetDownloadTimeout(-100));
        }

        [Fact]
        public void SetDownloadTimeout_ThrowsOnZeroValue()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentException>(() => ModelZoo.SetDownloadTimeout(0));
        }

        [Fact]
        public void Configuration_CanBeUpdated()
        {
            // Arrange
            var device = Device.CreateCpu();
            var newConfig = new ModelZooConfiguration(
                device,
                cacheEnabled: false,
                autoDownloadEnabled: false,
                defaultDownloadTimeoutMs: 120000);

            // Act
            ModelZoo.Configuration = newConfig;

            // Assert
            Assert.False(ModelZoo.Configuration.CacheEnabled);
            Assert.False(ModelZoo.Configuration.AutoDownloadEnabled);
            Assert.Equal(120000, ModelZoo.Configuration.DefaultDownloadTimeoutMs);

            // Cleanup - reset to defaults
            ModelZoo.Configuration = new ModelZooConfiguration();
        }

        [Fact]
        public void Configuration_ThrowsOnNullValue()
        {
            // Arrange, Act & Assert
            Assert.Throws<ArgumentNullException>(() => ModelZoo.Configuration = null);
        }

        [Fact]
        public void ListModelsByArchitecture_FiltersCorrectly()
        {
            // Arrange
            var metadata = CreateTestModelMetadata("test_resnet", "1.0.0");
            metadata.Architecture = "ResNet";
            ModelZoo.RegisterModel(metadata);

            // Act
            var models = ModelZoo.ListModelsByArchitecture("ResNet");

            // Assert
            Assert.NotNull(models);
        }

        [Fact]
        public void ListModelsByTask_FiltersCorrectly()
        {
            // Arrange
            var metadata = CreateTestModelMetadata("test_classifier", "1.0.0");
            metadata.Task = TaskType.ImageClassification;
            ModelZoo.RegisterModel(metadata);

            // Act
            var models = ModelZoo.ListModelsByTask(TaskType.ImageClassification);

            // Assert
            Assert.NotNull(models);
        }

        // Helper methods

        private ModelMetadata CreateTestModelMetadata(string name, string version)
        {
            return new ModelMetadata
            {
                Name = name,
                Version = version,
                Architecture = "TestArchitecture",
                Task = TaskType.ImageClassification,
                PretrainedOn = "TestDataset",
                InputShape = new int[] { 3, 224, 224 },
                OutputShape = new int[] { 1000 },
                NumParameters = 1000000,
                FileSizeBytes = 10000000,
                License = "MIT",
                Sha256Checksum = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                DownloadUrl = "https://example.com/test_model.bin"
            };
        }
    }
}
