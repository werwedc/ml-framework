using MLFramework.ModelVersioning;
using Xunit;

namespace MLFramework.Tests.ModelVersioning
{
    /// <summary>
    /// Unit tests for ModelRegistry.
    /// </summary>
    public class ModelRegistryTests
    {
        private readonly ModelRegistry _registry;

        public ModelRegistryTests()
        {
            _registry = new ModelRegistry();
        }

        #region RegisterModel Tests

        [Fact]
        public void RegisterModel_WithValidMetadata_ReturnsModelId()
        {
            // Arrange
            var metadata = new ModelMetadata
            {
                ModelName = "TestModel",
                Framework = "PyTorch",
                Architecture = "ResNet50"
            };

            // Act
            string modelId = _registry.RegisterModel("/path/to/model.pt", metadata);

            // Assert
            Assert.NotNull(modelId);
            Assert.NotEmpty(modelId);

            var modelInfo = _registry.GetModelById(modelId);
            Assert.NotNull(modelInfo);
            Assert.Equal(modelId, modelInfo!.ModelId);
            Assert.Equal("/path/to/model.pt", modelInfo.ModelPath);
            Assert.Equal("TestModel", modelInfo.Metadata.ModelName);
            Assert.Equal(LifecycleState.Draft, modelInfo.State);
        }

        [Fact]
        public void RegisterModel_WithNullMetadata_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                _registry.RegisterModel("/path/to/model.pt", null!));
        }

        [Fact]
        public void RegisterModel_WithEmptyPath_ThrowsArgumentException()
        {
            // Arrange
            var metadata = new ModelMetadata { ModelName = "TestModel" };

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _registry.RegisterModel("", metadata));

            Assert.Throws<ArgumentException>(() =>
                _registry.RegisterModel("   ", metadata));
        }

        [Fact]
        public void RegisterModel_GeneratesUniqueIds()
        {
            // Arrange
            var metadata = new ModelMetadata { ModelName = "TestModel" };

            // Act
            string id1 = _registry.RegisterModel("/path1/model.pt", metadata);
            string id2 = _registry.RegisterModel("/path2/model.pt", metadata);

            // Assert
            Assert.NotEqual(id1, id2);
        }

        #endregion

        #region TagModel Tests

        [Fact]
        public void TagModel_WithValidVersionTag_Succeeds()
        {
            // Arrange
            var metadata = new ModelMetadata { ModelName = "TestModel" };
            string modelId = _registry.RegisterModel("/path/to/model.pt", metadata);

            // Act
            _registry.TagModel(modelId, "v1.0.0");

            // Assert
            var modelInfo = _registry.GetModelById(modelId);
            Assert.NotNull(modelInfo);
            Assert.Equal("v1.0.0", modelInfo!.VersionTag);

            var retrievedByTag = _registry.GetModel("v1.0.0");
            Assert.Equal(modelId, retrievedByTag!.ModelId);
        }

        [Theory]
        [InlineData("v1.0.0")]
        [InlineData("v2.10.3")]
        [InlineData("v0.0.1")]
        public void TagModel_WithValidSemanticVersions_Succeeds(string versionTag)
        {
            // Arrange
            var metadata = new ModelMetadata { ModelName = "TestModel" };
            string modelId = _registry.RegisterModel("/path/to/model.pt", metadata);

            // Act & Assert
            _registry.TagModel(modelId, versionTag);
        }

        [Theory]
        [InlineData("1.0.0")]
        [InlineData("v1.0")]
        [InlineData("v1.0.0.0")]
        [InlineData("version-1.0.0")]
        [InlineData("")]
        [InlineData("   ")]
        public void TagModel_WithInvalidVersionTag_ThrowsArgumentException(string invalidTag)
        {
            // Arrange
            var metadata = new ModelMetadata { ModelName = "TestModel" };
            string modelId = _registry.RegisterModel("/path/to/model.pt", metadata);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _registry.TagModel(modelId, invalidTag));
        }

        [Fact]
        public void TagModel_WithNonExistentModel_ThrowsKeyNotFoundException()
        {
            // Act & Assert
            Assert.Throws<KeyNotFoundException>(() =>
                _registry.TagModel("non-existent-id", "v1.0.0"));
        }

        [Fact]
        public void TagModel_WithDuplicateTag_ThrowsInvalidOperationException()
        {
            // Arrange
            var metadata1 = new ModelMetadata { ModelName = "TestModel1" };
            var metadata2 = new ModelMetadata { ModelName = "TestModel2" };
            string id1 = _registry.RegisterModel("/path1/model.pt", metadata1);
            string id2 = _registry.RegisterModel("/path2/model.pt", metadata2);

            _registry.TagModel(id1, "v1.0.0");

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                _registry.TagModel(id2, "v1.0.0"));
        }

        #endregion

        #region GetModel Tests

        [Fact]
        public void GetModel_WithExistingVersionTag_ReturnsModelInfo()
        {
            // Arrange
            var metadata = new ModelMetadata { ModelName = "TestModel" };
            string modelId = _registry.RegisterModel("/path/to/model.pt", metadata);
            _registry.TagModel(modelId, "v1.0.0");

            // Act
            var modelInfo = _registry.GetModel("v1.0.0");

            // Assert
            Assert.NotNull(modelInfo);
            Assert.Equal(modelId, modelInfo!.ModelId);
            Assert.Equal("v1.0.0", modelInfo.VersionTag);
        }

        [Fact]
        public void GetModel_WithNonExistentTag_ReturnsNull()
        {
            // Act
            var modelInfo = _registry.GetModel("v99.99.99");

            // Assert
            Assert.Null(modelInfo);
        }

        #endregion

        #region GetModelById Tests

        [Fact]
        public void GetModelById_WithExistingId_ReturnsModelInfo()
        {
            // Arrange
            var metadata = new ModelMetadata { ModelName = "TestModel" };
            string modelId = _registry.RegisterModel("/path/to/model.pt", metadata);

            // Act
            var modelInfo = _registry.GetModelById(modelId);

            // Assert
            Assert.NotNull(modelInfo);
            Assert.Equal(modelId, modelInfo!.ModelId);
        }

        [Fact]
        public void GetModelById_WithNonExistentId_ReturnsNull()
        {
            // Act
            var modelInfo = _registry.GetModelById("non-existent-id");

            // Assert
            Assert.Null(modelInfo);
        }

        #endregion

        #region ListModels Tests

        [Fact]
        public void ListModels_ReturnsAllModels()
        {
            // Arrange
            var metadata1 = new ModelMetadata { ModelName = "TestModel1" };
            var metadata2 = new ModelMetadata { ModelName = "TestModel2" };
            var metadata3 = new ModelMetadata { ModelName = "TestModel3" };

            _registry.RegisterModel("/path1/model.pt", metadata1);
            Thread.Sleep(10); // Ensure different timestamps
            _registry.RegisterModel("/path2/model.pt", metadata2);
            Thread.Sleep(10);
            _registry.RegisterModel("/path3/model.pt", metadata3);

            // Act
            var models = _registry.ListModels().ToList();

            // Assert
            Assert.Equal(3, models.Count);
            Assert.All(models, m => Assert.NotNull(m));
        }

        [Fact]
        public void ListModels_OrdersByCreatedAt()
        {
            // Arrange
            var metadata1 = new ModelMetadata { ModelName = "TestModel1" };
            var metadata2 = new ModelMetadata { ModelName = "TestModel2" };

            _registry.RegisterModel("/path1/model.pt", metadata1);
            Thread.Sleep(10);
            string id2 = _registry.RegisterModel("/path2/model.pt", metadata2);

            // Act
            var models = _registry.ListModels().ToList();

            // Assert
            Assert.Equal(2, models.Count);
            Assert.Equal(id2, models[1].ModelId);
        }

        [Fact]
        public void ListModels_WithStateFilter_ReturnsMatchingModels()
        {
            // Arrange
            var metadata1 = new ModelMetadata { ModelName = "TestModel1" };
            var metadata2 = new ModelMetadata { ModelName = "TestModel2" };
            var metadata3 = new ModelMetadata { ModelName = "TestModel3" };

            string id1 = _registry.RegisterModel("/path1/model.pt", metadata1);
            string id2 = _registry.RegisterModel("/path2/model.pt", metadata2);
            string id3 = _registry.RegisterModel("/path3/model.pt", metadata3);

            _registry.UpdateModelState(id1, LifecycleState.Staging);
            _registry.UpdateModelState(id2, LifecycleState.Production);
            // id3 remains Draft

            // Act
            var draftModels = _registry.ListModels(LifecycleState.Draft).ToList();
            var stagingModels = _registry.ListModels(LifecycleState.Staging).ToList();
            var productionModels = _registry.ListModels(LifecycleState.Production).ToList();

            // Assert
            Assert.Single(draftModels);
            Assert.Single(stagingModels);
            Assert.Single(productionModels);
            Assert.Equal(id1, stagingModels[0].ModelId);
            Assert.Equal(id2, productionModels[0].ModelId);
            Assert.Equal(id3, draftModels[0].ModelId);
        }

        #endregion

        #region UpdateModelState Tests

        [Fact]
        public void UpdateModelState_WithValidTransition_Succeeds()
        {
            // Arrange
            var metadata = new ModelMetadata { ModelName = "TestModel" };
            string modelId = _registry.RegisterModel("/path/to/model.pt", metadata);

            // Act
            _registry.UpdateModelState(modelId, LifecycleState.Staging);

            // Assert
            var modelInfo = _registry.GetModelById(modelId);
            Assert.Equal(LifecycleState.Staging, modelInfo!.State);
        }

        [Theory]
        [InlineData(LifecycleState.Draft, LifecycleState.Staging)]
        [InlineData(LifecycleState.Draft, LifecycleState.Archived)]
        [InlineData(LifecycleState.Staging, LifecycleState.Production)]
        [InlineData(LifecycleState.Staging, LifecycleState.Draft)]
        [InlineData(LifecycleState.Staging, LifecycleState.Archived)]
        [InlineData(LifecycleState.Production, LifecycleState.Staging)]
        [InlineData(LifecycleState.Production, LifecycleState.Archived)]
        public void UpdateModelState_WithValidTransitions_Succeeds(LifecycleState from, LifecycleState to)
        {
            // Arrange
            var metadata = new ModelMetadata { ModelName = "TestModel" };
            string modelId = _registry.RegisterModel("/path/to/model.pt", metadata);
            _registry.UpdateModelState(modelId, from);

            // Act
            _registry.UpdateModelState(modelId, to);

            // Assert
            var modelInfo = _registry.GetModelById(modelId);
            Assert.Equal(to, modelInfo!.State);
        }

        [Theory]
        [InlineData(LifecycleState.Draft, LifecycleState.Production)]
        [InlineData(LifecycleState.Staging, LifecycleState.Staging)]
        [InlineData(LifecycleState.Production, LifecycleState.Draft)]
        [InlineData(LifecycleState.Archived, LifecycleState.Draft)]
        public void UpdateModelState_WithInvalidTransition_ThrowsInvalidOperationException(LifecycleState from, LifecycleState to)
        {
            // Arrange
            var metadata = new ModelMetadata { ModelName = "TestModel" };
            string modelId = _registry.RegisterModel("/path/to/model.pt", metadata);
            _registry.UpdateModelState(modelId, from);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                _registry.UpdateModelState(modelId, to));
        }

        [Fact]
        public void UpdateModelState_WithNonExistentModel_ThrowsKeyNotFoundException()
        {
            // Act & Assert
            Assert.Throws<KeyNotFoundException>(() =>
                _registry.UpdateModelState("non-existent-id", LifecycleState.Staging));
        }

        #endregion

        #region SetParentModel Tests

        [Fact]
        public void SetParentModel_WithValidParent_Succeeds()
        {
            // Arrange
            var parentMetadata = new ModelMetadata { ModelName = "ParentModel" };
            var childMetadata = new ModelMetadata { ModelName = "ChildModel" };

            string parentId = _registry.RegisterModel("/path/parent.pt", parentMetadata);
            string childId = _registry.RegisterModel("/path/child.pt", childMetadata);

            // Act
            _registry.SetParentModel(childId, parentId);

            // Assert
            var childInfo = _registry.GetModelById(childId);
            Assert.NotNull(childInfo);
            Assert.Equal(parentId, childInfo!.ParentModelId);
        }

        [Fact]
        public void SetParentModel_WithNonExistentModel_ThrowsKeyNotFoundException()
        {
            // Arrange
            var metadata = new ModelMetadata { ModelName = "TestModel" };
            string modelId = _registry.RegisterModel("/path/to/model.pt", metadata);

            // Act & Assert
            Assert.Throws<KeyNotFoundException>(() =>
                _registry.SetParentModel(modelId, "non-existent-parent"));
        }

        [Fact]
        public void SetParentModel_WithNonExistentParent_ThrowsKeyNotFoundException()
        {
            // Arrange
            var metadata = new ModelMetadata { ModelName = "TestModel" };
            _registry.RegisterModel("/path/to/model.pt", metadata);

            // Act & Assert
            Assert.Throws<KeyNotFoundException>(() =>
                _registry.SetParentModel("non-existent-child", "non-existent-parent"));
        }

        [Fact]
        public void SetParentModel_WithSelfAsParent_ThrowsArgumentException()
        {
            // Arrange
            var metadata = new ModelMetadata { ModelName = "TestModel" };
            string modelId = _registry.RegisterModel("/path/to/model.pt", metadata);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _registry.SetParentModel(modelId, modelId));
        }

        #endregion

        #region Concurrent Registration Tests

        [Fact]
        public void ConcurrentRegistration_GeneratesUniqueIds()
        {
            // Arrange
            const int threadCount = 100;
            var ids = new ConcurrentBag<string>();
            var exceptions = new ConcurrentBag<Exception>();
            var barrier = new Barrier(threadCount);

            // Act
            var tasks = Enumerable.Range(0, threadCount).Select(i => Task.Run(() =>
            {
                barrier.SignalAndWait(); // Synchronize start
                try
                {
                    var metadata = new ModelMetadata { ModelName = $"TestModel{i}" };
                    string id = _registry.RegisterModel($"/path/model{i}.pt", metadata);
                    ids.Add(id);
                }
                catch (Exception ex)
                {
                    exceptions.Add(ex);
                }
            })).ToArray();

            Task.WaitAll(tasks);

            // Assert
            Assert.Empty(exceptions);
            Assert.Equal(threadCount, ids.Count);
            Assert.Equal(threadCount, ids.Distinct().Count());
        }

        #endregion

        #region Metadata Tests

        [Fact]
        public void RegisterModel_WithFullMetadata_PreservesAllFields()
        {
            // Arrange
            var metadata = new ModelMetadata
            {
                ModelName = "TestModel",
                Description = "Test model for unit tests",
                Framework = "PyTorch",
                Architecture = "ResNet50",
                InputShape = new[] { 3, 224, 224 },
                OutputShape = new[] { 1000 },
                CustomMetadata = new Dictionary<string, string>
                {
                    { "author", "test-user" },
                    { "version", "1.0" }
                }
            };

            // Act
            string modelId = _registry.RegisterModel("/path/to/model.pt", metadata);
            var modelInfo = _registry.GetModelById(modelId);

            // Assert
            Assert.NotNull(modelInfo);
            Assert.Equal("TestModel", modelInfo!.Metadata.ModelName);
            Assert.Equal("Test model for unit tests", modelInfo.Metadata.Description);
            Assert.Equal("PyTorch", modelInfo.Metadata.Framework);
            Assert.Equal("ResNet50", modelInfo.Metadata.Architecture);
            Assert.Equal(new[] { 3, 224, 224 }, modelInfo.Metadata.InputShape);
            Assert.Equal(new[] { 1000 }, modelInfo.Metadata.OutputShape);
            Assert.NotNull(modelInfo.Metadata.CustomMetadata);
            Assert.Equal("test-user", modelInfo.Metadata.CustomMetadata["author"]);
            Assert.Equal("1.0", modelInfo.Metadata.CustomMetadata["version"]);
        }

        #endregion
    }
}
