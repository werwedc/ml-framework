using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using MLFramework.ModelRegistry;

namespace MLFramework.Tests.ModelRegistry
{
    public class ModelRegistryTests
    {
        private readonly ModelRegistry _registry;

        public ModelRegistryTests()
        {
            _registry = new ModelRegistry();
        }

        private ModelMetadata CreateTestMetadata(string version, string artifactPath = "path")
        {
            return new ModelMetadata
            {
                Version = version,
                TrainingDate = DateTime.Now,
                Hyperparameters = new Dictionary<string, object>
                {
                    { "learning_rate", 0.001f },
                    { "batch_size", 32 }
                },
                PerformanceMetrics = new Dictionary<string, float>
                {
                    { "accuracy", 0.95f },
                    { "f1_score", 0.93f }
                },
                ArtifactPath = artifactPath
            };
        }

        [Fact]
        public void RegisterSingleModel_VerifyQueryable()
        {
            // Arrange
            var metadata = CreateTestMetadata("1.0.0");

            // Act
            _registry.RegisterModel("resnet50", "1.0.0", metadata);

            // Assert
            Assert.True(_registry.HasVersion("resnet50", "1.0.0"));
            var retrieved = _registry.GetMetadata("resnet50", "1.0.0");
            Assert.Equal("1.0.0", retrieved.Version);
            Assert.Equal("path", retrieved.ArtifactPath);
        }

        [Fact]
        public void RegisterMultipleVersions_VerifyQueryable()
        {
            // Arrange
            var metadata1 = CreateTestMetadata("1.0.0", "path1");
            var metadata2 = CreateTestMetadata("1.1.0", "path2");
            var metadata3 = CreateTestMetadata("2.0.0", "path3");

            // Act
            _registry.RegisterModel("resnet50", "1.0.0", metadata1);
            _registry.RegisterModel("resnet50", "1.1.0", metadata2);
            _registry.RegisterModel("resnet50", "2.0.0", metadata3);

            // Assert
            var versions = _registry.GetVersions("resnet50").ToList();
            Assert.Equal(3, versions.Count);
            Assert.Contains("1.0.0", versions);
            Assert.Contains("1.1.0", versions);
            Assert.Contains("2.0.0", versions);
        }

        [Fact]
        public void RegisterDuplicateVersion_ThrowsInvalidOperationException()
        {
            // Arrange
            var metadata1 = CreateTestMetadata("1.0.0", "path1");
            var metadata2 = CreateTestMetadata("1.0.0", "path2");

            // Act
            _registry.RegisterModel("resnet50", "1.0.0", metadata1);

            // Assert
            Assert.Throws<InvalidOperationException>(() =>
            {
                _registry.RegisterModel("resnet50", "1.0.0", metadata2);
            });
        }

        [Fact]
        public void UnregisterExistingModel_VerifyRemoved()
        {
            // Arrange
            var metadata = CreateTestMetadata("1.0.0");
            _registry.RegisterModel("resnet50", "1.0.0", metadata);

            // Act
            _registry.UnregisterModel("resnet50", "1.0.0");

            // Assert
            Assert.False(_registry.HasVersion("resnet50", "1.0.0"));
            var versions = _registry.GetVersions("resnet50");
            Assert.Empty(versions);
        }

        [Fact]
        public void QueryNonExistentModel_ReturnsEmptyList()
        {
            // Act
            var versions = _registry.GetVersions("nonexistent");

            // Assert
            Assert.Empty(versions);
        }

        [Fact]
        public void HasVersion_NonExistent_ReturnsFalse()
        {
            // Act & Assert
            Assert.False(_registry.HasVersion("resnet50", "1.0.0"));
        }

        [Fact]
        public void GetMetadata_NonExistent_ThrowsKeyNotFoundException()
        {
            // Act & Assert
            Assert.Throws<KeyNotFoundException>(() =>
            {
                _registry.GetMetadata("resnet50", "1.0.0");
            });
        }

        [Fact]
        public void UnregisterNonExistentVersion_ThrowsKeyNotFoundException()
        {
            // Act & Assert
            Assert.Throws<KeyNotFoundException>(() =>
            {
                _registry.UnregisterModel("resnet50", "1.0.0");
            });
        }

        [Fact]
        public void UnregisterNonExistentModel_ThrowsKeyNotFoundException()
        {
            // Arrange
            _registry.RegisterModel("resnet50", "1.0.0", CreateTestMetadata("1.0.0"));

            // Act & Assert
            Assert.Throws<KeyNotFoundException>(() =>
            {
                _registry.UnregisterModel("nonexistent", "1.0.0");
            });
        }

        [Fact]
        public void GetAllModelNames_ReturnsAllModels()
        {
            // Arrange
            _registry.RegisterModel("resnet50", "1.0.0", CreateTestMetadata("1.0.0"));
            _registry.RegisterModel("resnet50", "1.1.0", CreateTestMetadata("1.1.0"));
            _registry.RegisterModel("mobilenet", "1.0.0", CreateTestMetadata("1.0.0"));
            _registry.RegisterModel("mobilenet", "2.0.0", CreateTestMetadata("2.0.0"));
            _registry.RegisterModel("bert", "1.0.0", CreateTestMetadata("1.0.0"));

            // Act
            var modelNames = _registry.GetAllModelNames().ToList();

            // Assert
            Assert.Equal(3, modelNames.Count);
            Assert.Contains("resnet50", modelNames);
            Assert.Contains("mobilenet", modelNames);
            Assert.Contains("bert", modelNames);
        }

        [Fact]
        public async Task ConcurrentRegistration_With10Threads_AllSucceed()
        {
            // Arrange
            const int threadCount = 10;
            var tasks = new Task[threadCount];
            var exceptions = new List<Exception>();

            // Act
            for (int i = 0; i < threadCount; i++)
            {
                int versionNum = i;
                tasks[i] = Task.Run(() =>
                {
                    try
                    {
                        var metadata = CreateTestMetadata($"{versionNum}.0.0", $"path{versionNum}");
                        _registry.RegisterModel("concurrent_model", $"{versionNum}.0.0", metadata);
                    }
                    catch (Exception ex)
                    {
                        lock (exceptions)
                        {
                            exceptions.Add(ex);
                        }
                    }
                });
            }

            await Task.WhenAll(tasks);

            // Assert
            Assert.Empty(exceptions);
            var versions = _registry.GetVersions("concurrent_model").ToList();
            Assert.Equal(threadCount, versions.Count);
        }

        [Fact]
        public async Task ConcurrentQuery_PerformanceTest_1000QueriesUnder10ms()
        {
            // Arrange
            const int queryCount = 1000;
            for (int i = 0; i < 100; i++)
            {
                _registry.RegisterModel($"model{i}", $"{i}.0.0", CreateTestMetadata($"{i}.0.0"));
            }

            var stopwatch = Stopwatch.StartNew();

            // Act
            var tasks = Enumerable.Range(0, queryCount)
                .Select(i => Task.Run(() =>
                {
                    var modelName = $"model{i % 100}";
                    var version = $"{(i % 100)}.0.0";
                    _registry.HasVersion(modelName, version);
                }))
                .ToArray();

            await Task.WhenAll(tasks);

            stopwatch.Stop();

            // Assert
            Assert.True(stopwatch.ElapsedMilliseconds < 10,
                $"Expected 1000 queries to complete in < 10ms, but took {stopwatch.ElapsedMilliseconds}ms");
        }

        [Fact]
        public void RegisterModel_WithNullName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
            {
                _registry.RegisterModel(null!, "1.0.0", CreateTestMetadata("1.0.0"));
            });
        }

        [Fact]
        public void RegisterModel_WithWhitespaceName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
            {
                _registry.RegisterModel("   ", "1.0.0", CreateTestMetadata("1.0.0"));
            });
        }

        [Fact]
        public void RegisterModel_WithNullVersion_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
            {
                _registry.RegisterModel("resnet50", null!, CreateTestMetadata("1.0.0"));
            });
        }

        [Fact]
        public void RegisterModel_WithNullMetadata_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
            {
                _registry.RegisterModel("resnet50", "1.0.0", null!);
            });
        }

        [Fact]
        public void GetVersions_SortsByVersion()
        {
            // Arrange
            _registry.RegisterModel("resnet50", "2.0.0", CreateTestMetadata("2.0.0"));
            _registry.RegisterModel("resnet50", "1.0.0", CreateTestMetadata("1.0.0"));
            _registry.RegisterModel("resnet50", "1.1.0", CreateTestMetadata("1.1.0"));
            _registry.RegisterModel("resnet50", "3.0.0", CreateTestMetadata("3.0.0"));

            // Act
            var versions = _registry.GetVersions("resnet50").ToList();

            // Assert
            Assert.Equal(new[] { "1.0.0", "1.1.0", "2.0.0", "3.0.0" }, versions);
        }

        [Fact]
        public void UnregisterLastVersion_RemovesModelEntry()
        {
            // Arrange
            _registry.RegisterModel("resnet50", "1.0.0", CreateTestMetadata("1.0.0"));

            // Act
            _registry.UnregisterModel("resnet50", "1.0.0");

            // Assert
            var modelNames = _registry.GetAllModelNames().ToList();
            Assert.DoesNotContain("resnet50", modelNames);
        }

        [Fact]
        public void GetVersions_ReturnsNewList_ModificationsDontAffectRegistry()
        {
            // Arrange
            _registry.RegisterModel("resnet50", "1.0.0", CreateTestMetadata("1.0.0"));
            _registry.RegisterModel("resnet50", "1.1.0", CreateTestMetadata("1.1.0"));

            // Act
            var versions1 = _registry.GetVersions("resnet50").ToList();
            versions1.Add("fake_version");
            var versions2 = _registry.GetVersions("resnet50").ToList();

            // Assert
            Assert.Equal(2, versions2.Count);
            Assert.DoesNotContain("fake_version", versions2);
        }
    }
}
