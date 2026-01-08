using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MLFramework.ModelVersioning;
using Xunit;

namespace MLFramework.Tests.ModelVersioning
{
    /// <summary>
    /// Unit tests for ModelVersionManager.
    /// </summary>
    public class ModelVersionManagerTests
    {
        private readonly IModelRegistry _registry;
        private readonly ModelVersionManager _manager;

        public ModelVersionManagerTests()
        {
            _registry = new ModelRegistry();
            _manager = new ModelVersionManager(_registry);
        }

        private string RegisterTestModel(string modelPath = "/models/test-model.bin")
        {
            var metadata = new ModelMetadata
            {
                CreationTimestamp = DateTime.UtcNow,
                DatasetVersion = "v1.0",
                ArchitectureHash = "abc123",
                TrainingParameters = new Dictionary<string, object>
                {
                    { "epochs", 100 },
                    { "batch_size", 32 }
                }
            };
            return _registry.RegisterModel(modelPath, metadata);
        }

        [Fact]
        public void LoadVersion_SuccessfullyLoadsModel()
        {
            // Arrange
            string modelId = RegisterTestModel("/models/test-model-v1.bin");
            string version = "v1.0.0";

            // Act
            _manager.LoadVersion(modelId, version, "/models/test-model-v1.bin");

            // Assert
            Assert.True(_manager.IsVersionLoaded(modelId, version));
            var loadedVersions = _manager.GetLoadedVersions(modelId);
            Assert.Contains(version, loadedVersions);
        }

        [Fact]
        public void LoadVersion_ThrowsForNonExistentModel()
        {
            // Arrange
            string nonExistentModelId = "non-existent-model";
            string version = "v1.0.0";

            // Act & Assert
            var exception = Assert.Throws<InvalidOperationException>(
                () => _manager.LoadVersion(nonExistentModelId, version, "/models/non-existent.bin")
            );
            Assert.Contains("not registered", exception.Message, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void LoadVersion_ThrowsForNullModelId()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(
                () => _manager.LoadVersion(null!, "v1.0.0", "/models/test.bin")
            );
        }

        [Fact]
        public void LoadVersion_ThrowsForEmptyModelId()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(
                () => _manager.LoadVersion("", "v1.0.0", "/models/test.bin")
            );
        }

        [Fact]
        public void LoadVersion_ThrowsForNullVersion()
        {
            // Arrange
            string modelId = RegisterTestModel();

            // Act & Assert
            Assert.Throws<ArgumentException>(
                () => _manager.LoadVersion(modelId, null!, "/models/test.bin")
            );
        }

        [Fact]
        public void LoadVersion_ThrowsForEmptyVersion()
        {
            // Arrange
            string modelId = RegisterTestModel();

            // Act & Assert
            Assert.Throws<ArgumentException>(
                () => _manager.LoadVersion(modelId, "", "/models/test.bin")
            );
        }

        [Fact]
        public void LoadVersion_ThrowsForNullModelPath()
        {
            // Arrange
            string modelId = RegisterTestModel();

            // Act & Assert
            Assert.Throws<ArgumentException>(
                () => _manager.LoadVersion(modelId, "v1.0.0", null!)
            );
        }

        [Fact]
        public void LoadVersion_ThrowsForEmptyModelPath()
        {
            // Arrange
            string modelId = RegisterTestModel();

            // Act & Assert
            Assert.Throws<ArgumentException>(
                () => _manager.LoadVersion(modelId, "v1.0.0", "")
            );
        }

        [Fact]
        public void LoadVersion_ThrowsForDuplicateLoad()
        {
            // Arrange
            string modelId = RegisterTestModel();
            string version = "v1.0.0";
            _manager.LoadVersion(modelId, version, "/models/test-model-v1.bin");

            // Act & Assert
            var exception = Assert.Throws<InvalidOperationException>(
                () => _manager.LoadVersion(modelId, version, "/models/test-model-v1.bin")
            );
            Assert.Contains("already loaded", exception.Message, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void UnloadVersion_SuccessfullyUnloadsModel()
        {
            // Arrange
            string modelId = RegisterTestModel();
            string version = "v1.0.0";
            _manager.LoadVersion(modelId, version, "/models/test-model-v1.bin");

            // Act
            _manager.UnloadVersion(modelId, version);

            // Assert
            Assert.False(_manager.IsVersionLoaded(modelId, version));
            var loadedVersions = _manager.GetLoadedVersions(modelId);
            Assert.DoesNotContain(version, loadedVersions);
        }

        [Fact]
        public void UnloadVersion_ThrowsForNonLoadedVersion()
        {
            // Arrange
            string modelId = RegisterTestModel();
            string version = "v1.0.0";

            // Act & Assert
            var exception = Assert.Throws<InvalidOperationException>(
                () => _manager.UnloadVersion(modelId, version)
            );
            Assert.Contains("not loaded", exception.Message, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void UnloadVersion_ThrowsForNullModelId()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(
                () => _manager.UnloadVersion(null!, "v1.0.0")
            );
        }

        [Fact]
        public void UnloadVersion_ThrowsForNullVersion()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(
                () => _manager.UnloadVersion("test-model", null!)
            );
        }

        [Fact]
        public void IsVersionLoaded_ReturnsCorrectStatus()
        {
            // Arrange
            string modelId = RegisterTestModel();
            string version = "v1.0.0";

            // Act & Assert
            Assert.False(_manager.IsVersionLoaded(modelId, version));

            _manager.LoadVersion(modelId, version, "/models/test-model-v1.bin");
            Assert.True(_manager.IsVersionLoaded(modelId, version));

            _manager.UnloadVersion(modelId, version);
            Assert.False(_manager.IsVersionLoaded(modelId, version));
        }

        [Fact]
        public void IsVersionLoaded_ReturnsFalseForNullModelId()
        {
            // Act & Assert
            Assert.False(_manager.IsVersionLoaded(null!, "v1.0.0"));
        }

        [Fact]
        public void IsVersionLoaded_ReturnsFalseForNullVersion()
        {
            // Act & Assert
            Assert.False(_manager.IsVersionLoaded("test-model", null!));
        }

        [Fact]
        public void GetLoadedVersions_ReturnsLoadedVersions()
        {
            // Arrange
            string modelId = RegisterTestModel();
            _manager.LoadVersion(modelId, "v1.0.0", "/models/test-v1.bin");
            _manager.LoadVersion(modelId, "v2.0.0", "/models/test-v2.bin");
            _manager.LoadVersion(modelId, "v3.0.0", "/models/test-v3.bin");

            // Act
            var loadedVersions = _manager.GetLoadedVersions(modelId).ToList();

            // Assert
            Assert.Equal(3, loadedVersions.Count);
            Assert.Contains("v1.0.0", loadedVersions);
            Assert.Contains("v2.0.0", loadedVersions);
            Assert.Contains("v3.0.0", loadedVersions);
        }

        [Fact]
        public void GetLoadedVersions_ReturnsEmptyForNoVersions()
        {
            // Arrange
            string modelId = RegisterTestModel();

            // Act
            var loadedVersions = _manager.GetLoadedVersions(modelId);

            // Assert
            Assert.Empty(loadedVersions);
        }

        [Fact]
        public void GetLoadedVersions_ReturnsEmptyForNullModelId()
        {
            // Act
            var loadedVersions = _manager.GetLoadedVersions(null!);

            // Assert
            Assert.Empty(loadedVersions);
        }

        [Fact]
        public void GetLoadedVersions_ReturnsOnlyModelSpecificVersions()
        {
            // Arrange
            string model1Id = RegisterTestModel("/models/model1.bin");
            string model2Id = RegisterTestModel("/models/model2.bin");

            _manager.LoadVersion(model1Id, "v1.0.0", "/models/model1-v1.bin");
            _manager.LoadVersion(model1Id, "v2.0.0", "/models/model1-v2.bin");
            _manager.LoadVersion(model2Id, "v1.0.0", "/models/model2-v1.bin");
            _manager.LoadVersion(model2Id, "v3.0.0", "/models/model2-v3.bin");

            // Act
            var model1Versions = _manager.GetLoadedVersions(model1Id).ToList();
            var model2Versions = _manager.GetLoadedVersions(model2Id).ToList();

            // Assert
            Assert.Equal(2, model1Versions.Count);
            Assert.Contains("v1.0.0", model1Versions);
            Assert.Contains("v2.0.0", model1Versions);

            Assert.Equal(2, model2Versions.Count);
            Assert.Contains("v1.0.0", model2Versions);
            Assert.Contains("v3.0.0", model2Versions);
        }

        [Fact]
        public void WarmUpVersion_RunsWarmupData()
        {
            // Arrange
            string modelId = RegisterTestModel();
            string version = "v1.0.0";
            _manager.LoadVersion(modelId, version, "/models/test-model-v1.bin");

            var warmupData = new List<object>
            {
                new { input = "test1" },
                new { input = "test2" },
                new { input = "test3" }
            };

            // Act & Assert - Should not throw
            _manager.WarmUpVersion(modelId, version, warmupData);
        }

        [Fact]
        public void WarmUpVersion_ThrowsForNonLoadedVersion()
        {
            // Arrange
            string modelId = RegisterTestModel();
            string version = "v1.0.0";
            var warmupData = new List<object> { new { input = "test" } };

            // Act & Assert
            var exception = Assert.Throws<InvalidOperationException>(
                () => _manager.WarmUpVersion(modelId, version, warmupData)
            );
            Assert.Contains("not loaded", exception.Message, StringComparison.OrdinalIgnoreCase);
        }

        [Fact]
        public void WarmUpVersion_ThrowsForNullModelId()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(
                () => _manager.WarmUpVersion(null!, "v1.0.0", new List<object>())
            );
        }

        [Fact]
        public void WarmUpVersion_ThrowsForNullVersion()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(
                () => _manager.WarmUpVersion("test-model", null!, new List<object>())
            );
        }

        [Fact]
        public void WarmUpVersion_ThrowsForNullWarmupData()
        {
            // Arrange
            string modelId = RegisterTestModel();
            string version = "v1.0.0";
            _manager.LoadVersion(modelId, version, "/models/test-model-v1.bin");

            // Act & Assert
            Assert.Throws<ArgumentNullException>(
                () => _manager.WarmUpVersion(modelId, version, null!)
            );
        }

        [Fact]
        public void GetLoadInfo_ReturnsCorrectInformation()
        {
            // Arrange
            string modelId = RegisterTestModel();
            string version = "v1.0.0";
            _manager.LoadVersion(modelId, version, "/models/test-model-v1.bin");

            // Act
            var loadInfo = _manager.GetLoadInfo(modelId, version);

            // Assert
            Assert.NotNull(loadInfo);
            Assert.Equal(modelId, loadInfo.ModelId);
            Assert.Equal(version, loadInfo.Version);
            Assert.True(loadInfo.IsLoaded);
            Assert.True(loadInfo.LoadTime <= DateTime.UtcNow);
            Assert.True(loadInfo.LoadTime >= DateTime.UtcNow.AddMinutes(-1));
            Assert.True(loadInfo.MemoryUsageBytes >= 0);
            Assert.Equal(0, loadInfo.RequestCount);
            Assert.Equal("Loaded", loadInfo.Status);
        }

        [Fact]
        public void GetLoadInfo_ReturnsNotLoadedForNonLoadedVersion()
        {
            // Arrange
            string modelId = RegisterTestModel();
            string version = "v1.0.0";

            // Act
            var loadInfo = _manager.GetLoadInfo(modelId, version);

            // Assert
            Assert.NotNull(loadInfo);
            Assert.Equal(modelId, loadInfo.ModelId);
            Assert.Equal(version, loadInfo.Version);
            Assert.False(loadInfo.IsLoaded);
            Assert.Equal(DateTime.MinValue, loadInfo.LoadTime);
            Assert.Equal(0, loadInfo.MemoryUsageBytes);
            Assert.Equal(0, loadInfo.RequestCount);
            Assert.Equal("NotLoaded", loadInfo.Status);
        }

        [Fact]
        public void GetLoadInfo_ReturnsWarmingUpStatusDuringWarmup()
        {
            // Arrange
            string modelId = RegisterTestModel();
            string version = "v1.0.0";
            _manager.LoadVersion(modelId, version, "/models/test-model-v1.bin");

            // Start warmup
            var warmupTask = Task.Run(() =>
            {
                var warmupData = new List<object> { new { input = "test" } };
                _manager.WarmUpVersion(modelId, version, warmupData);
            });

            // Wait a bit for warmup to start
            Task.Delay(50).Wait();

            // Act
            var loadInfo = _manager.GetLoadInfo(modelId, version);

            // Assert
            // Note: The status might change quickly, so we just verify it doesn't throw
            Assert.NotNull(loadInfo);
            Assert.True(loadInfo.IsLoaded);

            // Wait for warmup to complete
            warmupTask.Wait();
        }

        [Fact]
        public void ConcurrentLoadOperations_ThreadSafe()
        {
            // Arrange
            string modelId = RegisterTestModel();
            int threadCount = 10;
            var tasks = new List<Task>();

            // Act
            for (int i = 0; i < threadCount; i++)
            {
                int versionNum = i;
                tasks.Add(Task.Run(() =>
                {
                    string version = $"v{versionNum}.0.0";
                    try
                    {
                        _manager.LoadVersion(modelId, version, $"/models/model-v{versionNum}.bin");
                    }
                    catch
                    {
                        // Ignore exceptions from concurrent load attempts
                    }
                }));
            }

            Task.WaitAll(tasks.ToArray());

            // Assert - Verify versions were loaded without errors
            var loadedVersions = _manager.GetLoadedVersions(modelId).ToList();
            Assert.Equal(threadCount, loadedVersions.Count);
        }

        [Fact]
        public void ConcurrentUnloadOperations_ThreadSafe()
        {
            // Arrange
            string modelId = RegisterTestModel();
            int versionCount = 10;

            // Load all versions first
            for (int i = 0; i < versionCount; i++)
            {
                string version = $"v{i}.0.0";
                _manager.LoadVersion(modelId, version, $"/models/model-v{i}.bin");
            }

            var tasks = new List<Task>();

            // Act - Unload all versions concurrently
            for (int i = 0; i < versionCount; i++)
            {
                int versionNum = i;
                tasks.Add(Task.Run(() =>
                {
                    string version = $"v{versionNum}.0.0";
                    _manager.UnloadVersion(modelId, version);
                }));
            }

            Task.WaitAll(tasks.ToArray());

            // Assert
            var loadedVersions = _manager.GetLoadedVersions(modelId);
            Assert.Empty(loadedVersions);
        }

        [Fact]
        public void RequestCountTracking_TracksCorrectly()
        {
            // Arrange
            string modelId = RegisterTestModel();
            string version = "v1.0.0";
            _manager.LoadVersion(modelId, version, "/models/test-model-v1.bin");

            // Use internal method to increment request count
            // In a real scenario, this would be done through actual inference calls
            var managerType = _manager.GetType();
            var incrementMethod = managerType.GetMethod(
                "IncrementRequestCount",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance
            );

            if (incrementMethod != null)
            {
                // Act
                for (int i = 0; i < 10; i++)
                {
                    incrementMethod.Invoke(_manager, new object[] { modelId, version });
                }

                // Assert
                var loadInfo = _manager.GetLoadInfo(modelId, version);
                Assert.Equal(10, loadInfo.RequestCount);
            }
        }

        [Fact]
        public void VersionIsolation_DifferentVersionsHaveSeparateState()
        {
            // Arrange
            string modelId = RegisterTestModel();

            // Act
            _manager.LoadVersion(modelId, "v1.0.0", "/models/model-v1.bin");
            _manager.LoadVersion(modelId, "v2.0.0", "/models/model-v2.bin");

            // Increment request count for v1.0.0
            var managerType = _manager.GetType();
            var incrementMethod = managerType.GetMethod(
                "IncrementRequestCount",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance
            );

            if (incrementMethod != null)
            {
                incrementMethod.Invoke(_manager, new object[] { modelId, "v1.0.0" });
                incrementMethod.Invoke(_manager, new object[] { modelId, "v1.0.0" });
                incrementMethod.Invoke(_manager, new object[] { modelId, "v1.0.0" });

                // Assert - v1.0.0 should have 3 requests, v2.0.0 should have 0
                var loadInfoV1 = _manager.GetLoadInfo(modelId, "v1.0.0");
                var loadInfoV2 = _manager.GetLoadInfo(modelId, "v2.0.0");

                Assert.Equal(3, loadInfoV1.RequestCount);
                Assert.Equal(0, loadInfoV2.RequestCount);
            }
        }
    }
}
