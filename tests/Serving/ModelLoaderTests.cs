using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Xunit;
using MLFramework.Serving.Deployment;

namespace MLFramework.Tests.Serving
{
    /// <summary>
    /// Unit tests for ModelLoader functionality.
    /// </summary>
    public class ModelLoaderTests : IDisposable
    {
        private readonly string _testModelDirectory;
        private readonly ModelLoader _loader;
        private readonly MockModelFactory _mockFactory;

        public ModelLoaderTests()
        {
            _testModelDirectory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
            Directory.CreateDirectory(_testModelDirectory);

            _mockFactory = new MockModelFactory();
            _loader = new ModelLoader(_mockFactory.CreateModel);
        }

        public void Dispose()
        {
            try
            {
                if (Directory.Exists(_testModelDirectory))
                {
                    Directory.Delete(_testModelDirectory, true);
                }
            }
            catch
            {
                // Ignore cleanup errors
            }
        }

        [Fact]
        public void Load_ValidPath_LoadsAndTracksModel()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const string version = "1.0.0";

            // Act
            var model = _loader.Load(modelPath, version);

            // Assert
            Assert.NotNull(model);
            Assert.Equal("test-model", model.Name);
            Assert.Equal(version, model.Version);
            Assert.True(_loader.IsLoaded("test-model", version));
            Assert.Single(_loader.GetLoadedModels());
        }

        [Fact]
        public void Load_NonExistentPath_ThrowsFileNotFoundException()
        {
            // Arrange
            const string nonExistentPath = "/non/existent/path/model.onnx";
            const string version = "1.0.0";

            // Act & Assert
            Assert.Throws<FileNotFoundException>(() => _loader.Load(nonExistentPath, version));
        }

        [Fact]
        public void Load_DuplicateVersion_ThrowsInvalidOperationException()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const string version = "1.0.0";
            _loader.Load(modelPath, version);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => _loader.Load(modelPath, version));
        }

        [Fact]
        public void Load_InvalidVersionFormat_ThrowsArgumentException()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const string invalidVersion = "invalid";

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _loader.Load(modelPath, invalidVersion));
        }

        [Fact]
        public void Load_EmptyVersion_ThrowsArgumentException()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const string emptyVersion = "";

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _loader.Load(modelPath, emptyVersion));
        }

        [Fact]
        public void Load_WhitespaceVersion_ThrowsArgumentException()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const string whitespaceVersion = "   ";

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _loader.Load(modelPath, whitespaceVersion));
        }

        [Fact]
        public async Task LoadAsync_ValidPath_LoadsAndTracksModel()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const string version = "1.0.0";

            // Act
            var model = await _loader.LoadAsync(modelPath, version);

            // Assert
            Assert.NotNull(model);
            Assert.Equal("test-model", model.Name);
            Assert.Equal(version, model.Version);
            Assert.True(_loader.IsLoaded("test-model", version));
            Assert.Single(_loader.GetLoadedModels());
        }

        [Fact]
        public async Task LoadAsync_WithCancellation_CancelsLoad()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const string version = "1.0.0";
            var cts = new CancellationTokenSource();
            cts.Cancel();

            // Act & Assert
            await Assert.ThrowsAsync<OperationCanceledException>(
                () => _loader.LoadAsync(modelPath, version, cts.Token));
        }

        [Fact]
        public async Task LoadAsync_DuplicateVersion_ThrowsInvalidOperationException()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const string version = "1.0.0";
            await _loader.LoadAsync(modelPath, version);

            // Act & Assert
            await Assert.ThrowsAsync<InvalidOperationException>(
                () => _loader.LoadAsync(modelPath, version));
        }

        [Fact]
        public void Unload_ExistingModel_RemovesFromTracking()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const string version = "1.0.0";
            var model = _loader.Load(modelPath, version);

            // Act
            _loader.Unload(model);

            // Assert
            Assert.False(_loader.IsLoaded("test-model", version));
            Assert.Empty(_loader.GetLoadedModels());
        }

        [Fact]
        public void Unload_NonExistingModel_ThrowsInvalidOperationException()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const string version = "1.0.0";
            var model = _loader.Load(modelPath, version);
            _loader.Unload(model);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => _loader.Unload(model));
        }

        [Fact]
        public void IsLoaded_ExistingModel_ReturnsTrue()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const string version = "1.0.0";
            _loader.Load(modelPath, version);

            // Act & Assert
            Assert.True(_loader.IsLoaded("test-model", version));
        }

        [Fact]
        public void IsLoaded_NonExistingModel_ReturnsFalse()
        {
            // Act & Assert
            Assert.False(_loader.IsLoaded("test-model", "1.0.0"));
        }

        [Fact]
        public void GetLoadedModels_MultipleModels_ReturnsAllLoaded()
        {
            // Arrange
            _loader.Load(CreateTestModelFile("model1"), "1.0.0");
            _loader.Load(CreateTestModelFile("model2"), "1.0.0");
            _loader.Load(CreateTestModelFile("model1"), "1.1.0");

            // Act
            var models = _loader.GetLoadedModels().ToList();

            // Assert
            Assert.Equal(3, models.Count);
            Assert.Contains(models, m => m.Name == "model1" && m.Version == "1.0.0");
            Assert.Contains(models, m => m.Name == "model1" && m.Version == "1.1.0");
            Assert.Contains(models, m => m.Name == "model2" && m.Version == "1.0.0");
        }

        [Fact]
        public async Task Model_InferAsync_ReturnsResult()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const string version = "1.0.0";
            var model = _loader.Load(modelPath, version);
            var input = new InferenceInput("test input data");

            // Act
            var result = await model.InferAsync(input);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.Success);
            Assert.NotNull(result.Data);
            Assert.True(result.InferenceTimeMs >= 0);
        }

        [Fact]
        public void Model_IsActive_CanBeSet()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const string version = "1.0.0";
            var model = _loader.Load(modelPath, version);

            // Act
            model.IsActive = false;

            // Assert
            Assert.False(model.IsActive);
        }

        [Fact]
        public void Model_LoadTime_IsSetOnLoad()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const string version = "1.0.0";
            var beforeLoad = DateTime.UtcNow;

            // Act
            var model = _loader.Load(modelPath, version);
            var afterLoad = DateTime.UtcNow;

            // Assert
            Assert.InRange(model.LoadTime, beforeLoad, afterLoad);
        }

        [Fact]
        public void ConcurrentLoad_DifferentModels_Succeeds()
        {
            // Arrange
            const int threadCount = 10;
            var paths = Enumerable.Range(0, threadCount)
                .Select(i => CreateTestModelFile($"model{i}"))
                .ToList();

            // Act
            Parallel.For(0, threadCount, i =>
            {
                _loader.Load(paths[i], "1.0.0");
            });

            // Assert
            Assert.Equal(threadCount, _loader.GetLoadedModels().Count());
        }

        [Fact]
        public void ConcurrentLoad_SameModelDifferentVersions_Succeeds()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const int versionCount = 10;

            // Act
            Parallel.For(0, versionCount, i =>
            {
                _loader.Load(modelPath, $"1.0.{i}");
            });

            // Assert
            Assert.Equal(versionCount, _loader.GetLoadedModels().Count());
        }

        [Fact]
        public async Task ConcurrentAsyncLoad_DifferentModels_Succeeds()
        {
            // Arrange
            const int threadCount = 10;
            var paths = Enumerable.Range(0, threadCount)
                .Select(i => CreateTestModelFile($"model{i}"))
                .ToList();

            // Act
            var loadTasks = paths.Select(path => _loader.LoadAsync(path, "1.0.0"));
            await Task.WhenAll(loadTasks);

            // Assert
            Assert.Equal(threadCount, _loader.GetLoadedModels().Count());
        }

        [Fact]
        public void Model_Dispose_ReleasesResources()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const string version = "1.0.0";
            var model = _loader.Load(modelPath, version);
            var mockModel = (MockModel)model;

            // Act
            model.Dispose();

            // Assert
            Assert.True(mockModel.IsDisposed);
        }

        [Fact]
        public async Task Performance_Load_UnderTargetTime()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const string version = "1.0.0";
            const int iterations = 100;

            // Act
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                var model = _loader.Load(modelPath, version);
                _loader.Unload(model);
            }
            stopwatch.Stop();

            // Assert
            var avgTime = stopwatch.Elapsed.TotalMilliseconds / iterations;
            Assert.True(avgTime < 100, $"Average load time {avgTime}ms exceeds target 100ms");
        }

        [Fact]
        public async Task Performance_LoadAsync_UnderTargetTime()
        {
            // Arrange
            var modelPath = CreateTestModelFile("test-model");
            const string version = "1.0.0";
            const int iterations = 100;

            // Act
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                var model = await _loader.LoadAsync(modelPath, version);
                _loader.Unload(model);
            }
            stopwatch.Stop();

            // Assert
            var avgTime = stopwatch.Elapsed.TotalMilliseconds / iterations;
            Assert.True(avgTime < 10, $"Average async load overhead {avgTime}ms exceeds target 10ms");
        }

        // Helper methods

        private string CreateTestModelFile(string modelName)
        {
            var modelDir = Path.Combine(_testModelDirectory, modelName);
            Directory.CreateDirectory(modelDir);
            return modelDir;
        }

        // Mock model factory for testing

        private class MockModelFactory
        {
            private int _createCount = 0;

            public IModel CreateModel(string modelPath, string version)
            {
                var modelName = Path.GetFileName(modelPath);
                return new MockModel(modelName, version, _createCount++);
            }
        }

        private class MockModel : BaseModel
        {
            private int _instanceId;
            private bool _disposed = false;

            public MockModel(string name, string version, int instanceId = 0)
                : base(name, version)
            {
                _instanceId = instanceId;
            }

            public bool IsDisposed => _disposed;

            public override async Task<InferenceResult> InferAsync(InferenceInput input)
            {
                // Simulate inference latency
                await Task.Delay(1);

                // Return mock result
                return new InferenceResult(new
                {
                    InstanceId = _instanceId,
                    Name = Name,
                    Version = Version,
                    Input = input.Data,
                    Timestamp = DateTime.UtcNow
                })
                {
                    Success = true,
                    InferenceTimeMs = 1
                };
            }

            protected override void DisposeManagedResources()
            {
                if (!_disposed)
                {
                    _disposed = true;
                    base.DisposeManagedResources();
                }
            }
        }
    }
}
