using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;
using MLFramework.ModelZoo;

namespace MLFramework.Tests.ModelZooTests
{
    public class ModelRegistryTests : IDisposable
    {
        private readonly string _testDataPath;
        private readonly ModelRegistry _registry;

        public ModelRegistryTests()
        {
            _testDataPath = Path.Combine(Path.GetTempPath(), $"ModelRegistryTests_{Guid.NewGuid()}");
            Directory.CreateDirectory(_testDataPath);
            _registry = new ModelRegistry();
        }

        public void Dispose()
        {
            _registry.Clear();
            if (Directory.Exists(_testDataPath))
            {
                try
                {
                    Directory.Delete(_testDataPath, true);
                }
                catch
                {
                    // Ignore cleanup errors
                }
            }
        }

        #region Helper Methods

        private ModelMetadata CreateMetadata(string name, string version, string architecture, TaskType task)
        {
            return new ModelMetadata
            {
                Name = name,
                Version = version,
                Architecture = architecture,
                Task = task,
                PretrainedOn = "TestDataset",
                InputShape = new[] { 3, 224, 224 },
                OutputShape = new[] { 1000 },
                License = "MIT",
                Sha256Checksum = "abc123",
                DownloadUrl = "http://example.com/model.bin"
            };
        }

        private string CreateTestJsonFile(string fileName, ModelMetadata metadata)
        {
            var filePath = Path.Combine(_testDataPath, fileName);
            metadata.SaveToJsonFile(filePath);
            return filePath;
        }

        #endregion

        #region Register and Retrieve Models Tests

        [Fact]
        public void Register_ValidMetadata_SuccessfullyRegisters()
        {
            // Arrange
            var metadata = CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification);

            // Act
            _registry.Register(metadata);

            // Assert
            Assert.Equal(1, _registry.Count);
            var retrieved = _registry.Get("ResNet", "1.0.0");
            Assert.NotNull(retrieved);
            Assert.Equal("ResNet", retrieved!.Name);
            Assert.Equal("1.0.0", retrieved!.Version);
        }

        [Fact]
        public void Register_NullMetadata_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => _registry.Register(null!));
        }

        [Fact]
        public void Register_MetadataWithEmptyName_ThrowsArgumentException()
        {
            // Arrange
            var metadata = CreateMetadata("", "1.0.0", "ResNet", TaskType.ImageClassification);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _registry.Register(metadata));
        }

        [Fact]
        public void Register_MetadataWithEmptyVersion_ThrowsArgumentException()
        {
            // Arrange
            var metadata = CreateMetadata("ResNet", "", "ResNet", TaskType.ImageClassification);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _registry.Register(metadata));
        }

        [Fact]
        public void Get_RegisteredModel_ReturnsCorrectMetadata()
        {
            // Arrange
            var metadata = CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification);
            _registry.Register(metadata);

            // Act
            var retrieved = _registry.Get("ResNet", "1.0.0");

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal("ResNet", retrieved!.Name);
            Assert.Equal("1.0.0", retrieved!.Version);
            Assert.Equal("ResNet", retrieved!.Architecture);
            Assert.Equal(TaskType.ImageClassification, retrieved!.Task);
        }

        [Fact]
        public void Get_NonExistentModel_ReturnsNull()
        {
            // Act
            var retrieved = _registry.Get("NonExistent", "1.0.0");

            // Assert
            Assert.Null(retrieved);
        }

        [Fact]
        public void Get_NullName_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => _registry.Get(null!));
        }

        #endregion

        #region Version Resolution Tests

        [Fact]
        public void GetLatestVersion_SingleVersion_ReturnsThatVersion()
        {
            // Arrange
            var metadata = CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification);
            _registry.Register(metadata);

            // Act
            var latest = _registry.GetLatestVersion("ResNet");

            // Assert
            Assert.NotNull(latest);
            Assert.Equal("1.0.0", latest!.Version);
        }

        [Fact]
        public void GetLatestVersion_MultipleVersions_ReturnsHighestVersion()
        {
            // Arrange
            _registry.Register(CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification));
            _registry.Register(CreateMetadata("ResNet", "1.1.0", "ResNet", TaskType.ImageClassification));
            _registry.Register(CreateMetadata("ResNet", "2.0.0", "ResNet", TaskType.ImageClassification));
            _registry.Register(CreateMetadata("ResNet", "1.0.1", "ResNet", TaskType.ImageClassification));

            // Act
            var latest = _registry.GetLatestVersion("ResNet");

            // Assert
            Assert.NotNull(latest);
            Assert.Equal("2.0.0", latest!.Version);
        }

        [Fact]
        public void Get_NullVersion_ReturnsLatestVersion()
        {
            // Arrange
            _registry.Register(CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification));
            _registry.Register(CreateMetadata("ResNet", "2.0.0", "ResNet", TaskType.ImageClassification));

            // Act
            var retrieved = _registry.Get("ResNet");

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal("2.0.0", retrieved!.Version);
        }

        [Fact]
        public void GetLatestVersion_NonExistentModel_ReturnsNull()
        {
            // Act
            var latest = _registry.GetLatestVersion("NonExistent");

            // Assert
            Assert.Null(latest);
        }

        #endregion

        #region Filtering Tests

        [Fact]
        public void ListByArchitecture_MatchingArchitecture_ReturnsFilteredModels()
        {
            // Arrange
            _registry.Register(CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification));
            _registry.Register(CreateMetadata("ResNet", "1.1.0", "ResNet", TaskType.ImageClassification));
            _registry.Register(CreateMetadata("BERT", "1.0.0", "BERT", TaskType.TextClassification));

            // Act
            var resnetModels = _registry.ListByArchitecture("ResNet");

            // Assert
            Assert.Equal(2, resnetModels.Count);
            Assert.All(resnetModels, m => Assert.Equal("ResNet", m.Architecture));
        }

        [Fact]
        public void ListByArchitecture_CaseInsensitive_ReturnsFilteredModels()
        {
            // Arrange
            _registry.Register(CreateMetadata("ResNet", "1.0.0", "resnet", TaskType.ImageClassification));

            // Act
            var models = _registry.ListByArchitecture("RESNET");

            // Assert
            Assert.Single(models);
            Assert.Equal("resnet", models[0].Architecture);
        }

        [Fact]
        public void ListByArchitecture_NullArchitecture_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => _registry.ListByArchitecture(null!));
        }

        [Fact]
        public void ListByTask_MatchingTask_ReturnsFilteredModels()
        {
            // Arrange
            _registry.Register(CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification));
            _registry.Register(CreateMetadata("ResNet", "1.1.0", "ResNet", TaskType.ImageClassification));
            _registry.Register(CreateMetadata("BERT", "1.0.0", "BERT", TaskType.TextClassification));

            // Act
            var classificationModels = _registry.ListByTask(TaskType.ImageClassification);

            // Assert
            Assert.Equal(2, classificationModels.Count);
            Assert.All(classificationModels, m => Assert.Equal(TaskType.ImageClassification, m.Task));
        }

        [Fact]
        public void ListAll_ReturnsAllModels()
        {
            // Arrange
            _registry.Register(CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification));
            _registry.Register(CreateMetadata("BERT", "1.0.0", "BERT", TaskType.TextClassification));
            _registry.Register(CreateMetadata("YOLO", "1.0.0", "YOLO", TaskType.ObjectDetection));

            // Act
            var allModels = _registry.ListAll();

            // Assert
            Assert.Equal(3, allModels.Count);
        }

        #endregion

        #region Exists Tests

        [Fact]
        public void Exists_RegisteredModel_ReturnsTrue()
        {
            // Arrange
            _registry.Register(CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification));

            // Act
            bool exists = _registry.Exists("ResNet", "1.0.0");

            // Assert
            Assert.True(exists);
        }

        [Fact]
        public void Exists_NonExistentModel_ReturnsFalse()
        {
            // Act
            bool exists = _registry.Exists("NonExistent", "1.0.0");

            // Assert
            Assert.False(exists);
        }

        [Fact]
        public void Exists_WithNullVersion_ReturnsTrueIfAnyVersionExists()
        {
            // Arrange
            _registry.Register(CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification));

            // Act
            bool exists = _registry.Exists("ResNet");

            // Assert
            Assert.True(exists);
        }

        [Fact]
        public void Exists_NullName_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => _registry.Exists(null!));
        }

        #endregion

        #region Remove Tests

        [Fact]
        public void Remove_ExistingModel_SuccessfullyRemoves()
        {
            // Arrange
            _registry.Register(CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification));

            // Act
            bool removed = _registry.Remove("ResNet", "1.0.0");

            // Assert
            Assert.True(removed);
            Assert.Equal(0, _registry.Count);
            Assert.False(_registry.Exists("ResNet", "1.0.0"));
        }

        [Fact]
        public void Remove_NonExistentModel_ReturnsFalse()
        {
            // Act
            bool removed = _registry.Remove("NonExistent", "1.0.0");

            // Assert
            Assert.False(removed);
        }

        [Fact]
        public void Remove_WithNullVersion_RemovesAllVersions()
        {
            // Arrange
            _registry.Register(CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification));
            _registry.Register(CreateMetadata("ResNet", "2.0.0", "ResNet", TaskType.ImageClassification));

            // Act
            bool removed = _registry.Remove("ResNet");

            // Assert
            Assert.True(removed);
            Assert.Equal(0, _registry.Count);
            Assert.False(_registry.Exists("ResNet"));
        }

        [Fact]
        public void Remove_NullName_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => _registry.Remove(null!));
        }

        #endregion

        #region Persistence Tests

        [Fact]
        public void SaveToJson_WritesValidJsonFile()
        {
            // Arrange
            _registry.Register(CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification));
            _registry.Register(CreateMetadata("BERT", "1.0.0", "BERT", TaskType.TextClassification));
            var filePath = Path.Combine(_testDataPath, "registry.json");

            // Act
            _registry.SaveToJson(filePath);

            // Assert
            Assert.True(File.Exists(filePath));
            var json = File.ReadAllText(filePath);
            Assert.NotEmpty(json);
            Assert.Contains("ResNet", json);
            Assert.Contains("BERT", json);
        }

        [Fact]
        public void LoadFromJson_ValidJson_SuccessfullyLoads()
        {
            // Arrange
            var metadata1 = CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification);
            var metadata2 = CreateMetadata("BERT", "1.0.0", "BERT", TaskType.TextClassification);
            var filePath = Path.Combine(_testDataPath, "registry.json");

            // Create and save a registry
            var originalRegistry = new ModelRegistry();
            originalRegistry.Register(metadata1);
            originalRegistry.Register(metadata2);
            originalRegistry.SaveToJson(filePath);

            // Act
            _registry.LoadFromJson(filePath);

            // Assert
            Assert.Equal(2, _registry.Count);
            Assert.True(_registry.Exists("ResNet", "1.0.0"));
            Assert.True(_registry.Exists("BERT", "1.0.0"));
        }

        [Fact]
        public void LoadFromJson_NonExistentFile_ThrowsFileNotFoundException()
        {
            // Act & Assert
            Assert.Throws<FileNotFoundException>(() => _registry.LoadFromJson("nonexistent.json"));
        }

        [Fact]
        public void LoadFromJson_NullFilePath_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => _registry.LoadFromJson(null!));
        }

        [Fact]
        public void SaveAndLoad_RoundTrip_PreservesData()
        {
            // Arrange
            var metadata1 = CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification);
            var metadata2 = CreateMetadata("BERT", "2.0.0", "BERT", TaskType.TextClassification);
            var filePath = Path.Combine(_testDataPath, "roundtrip.json");

            _registry.Register(metadata1);
            _registry.Register(metadata2);
            _registry.SaveToJson(filePath);

            // Act
            var newRegistry = new ModelRegistry();
            newRegistry.LoadFromJson(filePath);

            // Assert
            Assert.Equal(_registry.Count, newRegistry.Count);
            var originalModel1 = _registry.Get("ResNet", "1.0.0");
            var loadedModel1 = newRegistry.Get("ResNet", "1.0.0");
            Assert.Equal(originalModel1!.Name, loadedModel1!.Name);
            Assert.Equal(originalModel1.Version, loadedModel1.Version);
            Assert.Equal(originalModel1.Architecture, loadedModel1.Architecture);
            Assert.Equal(originalModel1.Task, loadedModel1.Task);
        }

        #endregion

        #region Clear Tests

        [Fact]
        public void_Clear_RemovesAllModels()
        {
            // Arrange
            _registry.Register(CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification));
            _registry.Register(CreateMetadata("BERT", "1.0.0", "BERT", TaskType.TextClassification));

            // Act
            _registry.Clear();

            // Assert
            Assert.Equal(0, _registry.Count);
            Assert.Empty(_registry.ListAll());
        }

        #endregion

        #region Thread Safety Tests

        [Fact]
        public void ConcurrentAccess_ThreadSafe()
        {
            // Arrange
            const int threadCount = 10;
            const int modelsPerThread = 100;

            // Act
            Parallel.For(0, threadCount, i =>
            {
                for (int j = 0; j < modelsPerThread; j++)
                {
                    var metadata = CreateMetadata(
                        $"Model_{i}_{j}",
                        "1.0.0",
                        "TestArch",
                        TaskType.ImageClassification);
                    _registry.Register(metadata);
                }
            });

            // Assert
            Assert.Equal(threadCount * modelsPerThread, _registry.Count);
        }

        [Fact]
        public void ConcurrentReadWrite_ThreadSafe()
        {
            // Arrange
            for (int i = 0; i < 100; i++)
            {
                _registry.Register(CreateMetadata($"Model_{i}", "1.0.0", "TestArch", TaskType.ImageClassification));
            }

            // Act
            var exceptions = new System.Collections.Concurrent.ConcurrentQueue<Exception>();
            Parallel.For(0, 10, i =>
            {
                try
                {
                    if (i % 2 == 0)
                    {
                        // Read operations
                        _ = _registry.ListAll();
                        _ = _registry.Get($"Model_{i % 100}");
                    }
                    else
                    {
                        // Write operations
                        _registry.Register(CreateMetadata($"NewModel_{i}", "1.0.0", "TestArch", TaskType.ImageClassification));
                        _registry.Remove($"Model_{i % 100}");
                    }
                }
                catch (Exception ex)
                {
                    exceptions.Enqueue(ex);
                }
            });

            // Assert
            Assert.Empty(exceptions);
        }

        #endregion
    }

    public class RegistryBuilderTests : IDisposable
    {
        private readonly string _testDataPath;

        public RegistryBuilderTests()
        {
            _testDataPath = Path.Combine(Path.GetTempPath(), $"RegistryBuilderTests_{Guid.NewGuid()}");
            Directory.CreateDirectory(_testDataPath);
        }

        public void Dispose()
        {
            if (Directory.Exists(_testDataPath))
            {
                try
                {
                    Directory.Delete(_testDataPath, true);
                }
                catch
                {
                    // Ignore cleanup errors
                }
            }
        }

        #region Helper Methods

        private ModelMetadata CreateMetadata(string name, string version, string architecture, TaskType task)
        {
            return new ModelMetadata
            {
                Name = name,
                Version = version,
                Architecture = architecture,
                Task = task,
                PretrainedOn = "TestDataset",
                InputShape = new[] { 3, 224, 224 },
                OutputShape = new[] { 1000 },
                License = "MIT",
                Sha256Checksum = "abc123",
                DownloadUrl = "http://example.com/model.bin"
            };
        }

        #endregion

        #region Add Tests

        [Fact]
        public void Add_ValidMetadata_SuccessfullyAdds()
        {
            // Arrange
            var builder = new RegistryBuilder();
            var metadata = CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification);

            // Act
            builder.Add(metadata);

            // Assert
            Assert.Equal(1, builder.Count);
        }

        [Fact]
        public void Add_NullMetadata_ThrowsArgumentNullException()
        {
            // Arrange
            var builder = new RegistryBuilder();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => builder.Add(null!));
        }

        [Fact]
        public void AddRange_ValidList_SuccessfullyAddsMultiple()
        {
            // Arrange
            var builder = new RegistryBuilder();
            var metadataList = new List<ModelMetadata>
            {
                CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification),
                CreateMetadata("BERT", "1.0.0", "BERT", TaskType.TextClassification),
                CreateMetadata("YOLO", "1.0.0", "YOLO", TaskType.ObjectDetection)
            };

            // Act
            builder.AddRange(metadataList);

            // Assert
            Assert.Equal(3, builder.Count);
        }

        [Fact]
        public void AddRange_NullList_ThrowsArgumentNullException()
        {
            // Arrange
            var builder = new RegistryBuilder();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => builder.AddRange(null!));
        }

        #endregion

        #region AddFromJsonFile Tests

        [Fact]
        public void AddFromJsonFile_ValidFile_SuccessfullyAdds()
        {
            // Arrange
            var builder = new RegistryBuilder();
            var metadata = CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification);
            var filePath = Path.Combine(_testDataPath, "test.json");
            metadata.SaveToJsonFile(filePath);

            // Act
            builder.AddFromJsonFile(filePath);

            // Assert
            Assert.Equal(1, builder.Count);
        }

        [Fact]
        public void AddFromJsonFile_NonExistentFile_ThrowsFileNotFoundException()
        {
            // Arrange
            var builder = new RegistryBuilder();

            // Act & Assert
            Assert.Throws<FileNotFoundException>(() => builder.AddFromJsonFile("nonexistent.json"));
        }

        [Fact]
        public void AddFromJsonFile_NullPath_ThrowsArgumentNullException()
        {
            // Arrange
            var builder = new RegistryBuilder();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => builder.AddFromJsonFile(null!));
        }

        #endregion

        #region AddFromDirectory Tests

        [Fact]
        public void AddFromDirectory_ContainsJsonFiles_SuccessfullyAddsAll()
        {
            // Arrange
            var builder = new RegistryBuilder();

            // Create test files
            CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification)
                .SaveToJsonFile(Path.Combine(_testDataPath, "resnet.json"));
            CreateMetadata("BERT", "1.0.0", "BERT", TaskType.TextClassification)
                .SaveToJsonFile(Path.Combine(_testDataPath, "bert.json"));
            CreateMetadata("YOLO", "1.0.0", "YOLO", TaskType.ObjectDetection)
                .SaveToJsonFile(Path.Combine(_testDataPath, "yolo.json"));

            // Act
            builder.AddFromDirectory(_testDataPath);

            // Assert
            Assert.Equal(3, builder.Count);
        }

        [Fact]
        public void AddFromDirectory_NonExistentDirectory_ThrowsDirectoryNotFoundException()
        {
            // Arrange
            var builder = new RegistryBuilder();

            // Act & Assert
            Assert.Throws<DirectoryNotFoundException>(() => builder.AddFromDirectory("nonexistent_directory"));
        }

        [Fact]
        public void AddFromDirectory_NullPath_ThrowsArgumentNullException()
        {
            // Arrange
            var builder = new RegistryBuilder();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => builder.AddFromDirectory(null!));
        }

        #endregion

        #region Build Tests

        [Fact]
        public void Build_WithMetadata_ReturnsPopulatedRegistry()
        {
            // Arrange
            var builder = new RegistryBuilder();
            builder.Add(CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification));
            builder.Add(CreateMetadata("BERT", "1.0.0", "BERT", TaskType.TextClassification));

            // Act
            var registry = builder.Build();

            // Assert
            Assert.NotNull(registry);
            Assert.Equal(2, registry.Count);
            Assert.True(registry.Exists("ResNet", "1.0.0"));
            Assert.True(registry.Exists("BERT", "1.0.0"));
        }

        [Fact]
        public void Build_WithDuplicateNames_UpdatesRegistration()
        {
            // Arrange
            var builder = new RegistryBuilder();
            builder.Add(CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification));
            builder.Add(CreateMetadata("ResNet", "1.0.0", "ResNetV2", TaskType.ImageClassification));

            // Act
            var registry = builder.Build();

            // Assert
            Assert.Equal(1, registry.Count);
            var retrieved = registry.Get("ResNet", "1.0.0");
            Assert.Equal("ResNetV2", retrieved!.Architecture);
        }

        [Fact]
        public void Build_EmptyBuilder_ReturnsEmptyRegistry()
        {
            // Arrange
            var builder = new RegistryBuilder();

            // Act
            var registry = builder.Build();

            // Assert
            Assert.NotNull(registry);
            Assert.Equal(0, registry.Count);
        }

        #endregion

        #region Clear Tests

        [Fact]
        public void Clear_RemovesAllMetadata()
        {
            // Arrange
            var builder = new RegistryBuilder();
            builder.Add(CreateMetadata("ResNet", "1.0.0", "ResNet", TaskType.ImageClassification));
            builder.Add(CreateMetadata("BERT", "1.0.0", "BERT", TaskType.TextClassification));

            // Act
            builder.Clear();

            // Assert
            Assert.Equal(0, builder.Count);
        }

        #endregion
    }
}
