using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using Microsoft.Extensions.Logging;
using MLFramework.Serving.Deployment;

namespace MLFramework.Tests.Serving.Routing
{
    /// <summary>
    /// Unit tests for VersionRouter functionality.
    /// </summary>
    public class VersionRouterTests : IDisposable
    {
        private readonly MockModelRegistry _mockRegistry;
        private readonly MockModelLoader _mockLoader;
        private readonly VersionRouter _router;

        public VersionRouterTests()
        {
            _mockRegistry = new MockModelRegistry();
            _mockLoader = new MockModelLoader();
            var logger = new ConsoleLogger();
            _router = new VersionRouter(_mockRegistry, _mockLoader, logger);
        }

        public void Dispose()
        {
            _mockLoader.ClearAllModels();
        }

        [Fact]
        public void GetModel_WithExplicitVersion_ReturnsCorrectModel()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "2.0.0";
            _mockRegistry.RegisterModel(modelName, version, new ModelMetadata());
            var expectedModel = _mockLoader.LoadModel(modelName, version);

            // Act
            var result = _router.GetModel(modelName, version);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(modelName, result.Name);
            Assert.Equal(version, result.Version);
            Assert.Same(expectedModel, result);
        }

        [Fact]
        public void GetModel_WithContextAndPreferredVersion_ReturnsCorrectModel()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "2.0.0";
            _mockRegistry.RegisterModel(modelName, version, new ModelMetadata());
            var expectedModel = _mockLoader.LoadModel(modelName, version);

            var context = new RoutingContext { PreferredVersion = version };

            // Act
            var result = _router.GetModel(modelName, context);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(version, result.Version);
            Assert.Same(expectedModel, result);
        }

        [Fact]
        public void GetModel_WithoutVersionAndNoDefault_UsesLatest()
        {
            // Arrange
            const string modelName = "test-model";
            _mockRegistry.RegisterModel(modelName, "1.0.0", new ModelMetadata());
            _mockRegistry.RegisterModel(modelName, "2.0.0", new ModelMetadata());
            _mockRegistry.RegisterModel(modelName, "1.1.0", new ModelMetadata());

            var model10 = _mockLoader.LoadModel(modelName, "1.0.0");
            var model20 = _mockLoader.LoadModel(modelName, "2.0.0");
            var model11 = _mockLoader.LoadModel(modelName, "1.1.0");

            var context = new RoutingContext();

            // Act
            var result = _router.GetModel(modelName, context);

            // Assert
            Assert.NotNull(result);
            Assert.Equal("2.0.0", result.Version); // Latest version
            Assert.Same(model20, result);
        }

        [Fact]
        public void GetModel_WithoutVersionAndWithDefault_UsesDefault()
        {
            // Arrange
            const string modelName = "test-model";
            _mockRegistry.RegisterModel(modelName, "1.0.0", new ModelMetadata());
            _mockRegistry.RegisterModel(modelName, "2.0.0", new ModelMetadata());
            _mockLoader.LoadModel(modelName, "1.0.0");
            _mockLoader.LoadModel(modelName, "2.0.0");

            _router.SetDefaultVersion(modelName, "1.0.0");
            var context = new RoutingContext();

            // Act
            var result = _router.GetModel(modelName, context);

            // Assert
            Assert.NotNull(result);
            Assert.Equal("1.0.0", result.Version);
        }

        [Fact]
        public void GetModel_ToNonExistentVersion_ThrowsRoutingException()
        {
            // Arrange
            const string modelName = "test-model";
            _mockRegistry.RegisterModel(modelName, "1.0.0", new ModelMetadata());
            _mockLoader.LoadModel(modelName, "1.0.0");

            // Act & Assert
            var ex = Assert.Throws<RoutingException>(() => _router.GetModel(modelName, "2.0.0"));
            Assert.Equal(modelName, ex.ModelName);
            Assert.Equal("2.0.0", ex.RequestedVersion);
            Assert.Contains("not found", ex.Message);
        }

        [Fact]
        public void GetModel_ModelNotLoaded_ThrowsRoutingException()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            _mockRegistry.RegisterModel(modelName, version, new ModelMetadata());
            // Don't load the model

            // Act & Assert
            var ex = Assert.Throws<RoutingException>(() => _router.GetModel(modelName, version));
            Assert.Contains("not loaded", ex.Message);
        }

        [Fact]
        public void GetModel_ModelWithNoVersions_ThrowsRoutingException()
        {
            // Arrange
            const string modelName = "empty-model";
            var context = new RoutingContext();

            // Act & Assert
            var ex = Assert.Throws<RoutingException>(() => _router.GetModel(modelName, context));
            Assert.Contains("No versions found", ex.Message);
        }

        [Fact]
        public void GetModel_WithNullModelName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => _router.GetModel(null!, new RoutingContext()));
        }

        [Fact]
        public void GetModel_WithEmptyModelName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => _router.GetModel("", new RoutingContext()));
        }

        [Fact]
        public void GetModel_WithNullVersion_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => _router.GetModel("test-model", (string?)null!));
        }

        [Fact]
        public void GetModel_WithEmptyVersion_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => _router.GetModel("test-model", ""));
        }

        [Fact]
        public void SetDefaultVersion_ValidVersion_SetsDefault()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            _mockRegistry.RegisterModel(modelName, version, new ModelMetadata());

            // Act
            _router.SetDefaultVersion(modelName, version);

            // Assert
            Assert.Equal(version, _router.GetDefaultVersion(modelName));
        }

        [Fact]
        public void SetDefaultVersion_ToNonExistentVersion_ThrowsKeyNotFoundException()
        {
            // Arrange
            const string modelName = "test-model";

            // Act & Assert
            Assert.Throws<KeyNotFoundException>(() =>
                _router.SetDefaultVersion(modelName, "1.0.0"));
        }

        [Fact]
        public void SetDefaultVersion_WithNullModelName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => _router.SetDefaultVersion(null!, "1.0.0"));
        }

        [Fact]
        public void SetDefaultVersion_WithEmptyModelName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => _router.SetDefaultVersion("", "1.0.0"));
        }

        [Fact]
        public void SetDefaultVersion_WithNullVersion_ThrowsArgumentException()
        {
            // Arrange
            const string modelName = "test-model";
            _mockRegistry.RegisterModel(modelName, "1.0.0", new ModelMetadata());

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _router.SetDefaultVersion(modelName, null!));
        }

        [Fact]
        public void SetDefaultVersion_WithEmptyVersion_ThrowsArgumentException()
        {
            // Arrange
            const string modelName = "test-model";
            _mockRegistry.RegisterModel(modelName, "1.0.0", new ModelMetadata());

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _router.SetDefaultVersion(modelName, ""));
        }

        [Fact]
        public void GetDefaultVersion_WhenNotSet_ReturnsNull()
        {
            // Arrange
            const string modelName = "test-model";

            // Act
            var result = _router.GetDefaultVersion(modelName);

            // Assert
            Assert.Null(result);
        }

        [Fact]
        public void GetDefaultVersion_WithNullModelName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => _router.GetDefaultVersion(null!));
        }

        [Fact]
        public void GetDefaultVersion_WithEmptyModelName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => _router.GetDefaultVersion(""));
        }

        [Fact]
        public void SemanticVersion_Parsing_ValidFormats()
        {
            // Arrange & Act
            var v1 = SemanticVersion.Parse("1.0.0");
            var v2 = SemanticVersion.Parse("2.1.3");
            var v3 = SemanticVersion.Parse("10.20.30");
            var v4 = SemanticVersion.Parse("1.0.0-beta");
            var v5 = SemanticVersion.Parse("1.0.0-beta.1");
            var v6 = SemanticVersion.Parse("1.0.0+build123");
            var v7 = SemanticVersion.Parse("1.0.0-beta.1+build123");

            // Assert
            Assert.Equal(1, v1.Major);
            Assert.Equal(0, v1.Minor);
            Assert.Equal(0, v1.Patch);

            Assert.Equal(2, v2.Major);
            Assert.Equal(1, v2.Minor);
            Assert.Equal(3, v2.Patch);

            Assert.Equal(10, v3.Major);
            Assert.Equal(20, v3.Minor);
            Assert.Equal(30, v3.Patch);

            Assert.Equal("beta", v4.PreRelease);
            Assert.Equal("beta.1", v5.PreRelease);
            Assert.Equal("build123", v6.BuildMetadata);
            Assert.Equal("beta.1", v7.PreRelease);
            Assert.Equal("build123", v7.BuildMetadata);
        }

        [Fact]
        public void SemanticVersion_Parsing_InvalidFormat_ThrowsFormatException()
        {
            // Act & Assert
            Assert.Throws<FormatException>(() => SemanticVersion.Parse("invalid"));
            Assert.Throws<FormatException>(() => SemanticVersion.Parse("1.0"));
            Assert.Throws<FormatException>(() => SemanticVersion.Parse("1.0.0.0"));
            Assert.Throws<FormatException>(() => SemanticVersion.Parse("1.0."));
            Assert.Throws<FormatException>(() => SemanticVersion.Parse(".1.0"));
        }

        [Fact]
        public void SemanticVersion_Comparison_WorksCorrectly()
        {
            // Arrange
            var v100 = SemanticVersion.Parse("1.0.0");
            var v110 = SemanticVersion.Parse("1.1.0");
            var v200 = SemanticVersion.Parse("2.0.0");
            var v100beta = SemanticVersion.Parse("1.0.0-beta");

            // Assert
            Assert.True(v200 > v110);
            Assert.True(v110 > v100);
            Assert.True(v100 > v100beta);
            Assert.False(v100 > v200);

            Assert.True(v100 < v110);
            Assert.True(v100beta < v100);

            Assert.True(v100 == new SemanticVersion(1, 0, 0));
            Assert.True(v100 == SemanticVersion.Parse("1.0.0"));

            Assert.Equal(0, v100.CompareTo(new SemanticVersion(1, 0, 0)));
            Assert.True(v100.CompareTo(v110) < 0);
            Assert.True(v200.CompareTo(v100) > 0);
        }

        [Fact]
        public void SemanticVersion_TryParse_ReturnsCorrectly()
        {
            // Act & Assert
            Assert.True(SemanticVersion.TryParse("1.0.0", out var v1));
            Assert.Equal(1, v1.Major);

            Assert.False(SemanticVersion.TryParse("invalid", out var v2));
            Assert.Equal(default, v2);
        }

        [Fact]
        public void GetModel_WithSemanticVersioning_SelectsLatest()
        {
            // Arrange
            const string modelName = "test-model";
            _mockRegistry.RegisterModel(modelName, "1.0.0", new ModelMetadata());
            _mockRegistry.RegisterModel(modelName, "1.2.3", new ModelMetadata());
            _mockRegistry.RegisterModel(modelName, "2.0.0", new ModelMetadata());
            _mockRegistry.RegisterModel(modelName, "1.10.0", new ModelMetadata());
            _mockRegistry.RegisterModel(modelName, "1.0.1", new ModelMetadata());

            // Load all versions
            _mockLoader.LoadModel(modelName, "1.0.0");
            _mockLoader.LoadModel(modelName, "1.2.3");
            var model20 = _mockLoader.LoadModel(modelName, "2.0.0");
            _mockLoader.LoadModel(modelName, "1.10.0");
            _mockLoader.LoadModel(modelName, "1.0.1");

            var context = new RoutingContext();

            // Act
            var result = _router.GetModel(modelName, context);

            // Assert
            Assert.NotNull(result);
            Assert.Equal("2.0.0", result.Version);
            Assert.Same(model20, result);
        }

        [Fact]
        public void GetModel_WithPreReleaseVersions_SelectsLatestStable()
        {
            // Arrange
            const string modelName = "test-model";
            _mockRegistry.RegisterModel(modelName, "2.0.0", new ModelMetadata());
            _mockRegistry.RegisterModel(modelName, "2.1.0-beta", new ModelMetadata());
            _mockRegistry.RegisterModel(modelName, "2.0.1", new ModelMetadata());

            var model20 = _mockLoader.LoadModel(modelName, "2.0.0");
            _mockLoader.LoadModel(modelName, "2.1.0-beta");
            _mockLoader.LoadModel(modelName, "2.0.1");

            var context = new RoutingContext();

            // Act
            var result = _router.GetModel(modelName, context);

            // Assert
            Assert.NotNull(result);
            Assert.Equal("2.0.1", result.Version); // Latest stable, not pre-release
        }

        [Fact]
        public void Performance_Routing_Under1ms()
        {
            // Arrange
            const string modelName = "test-model";
            _mockRegistry.RegisterModel(modelName, "1.0.0", new ModelMetadata());
            _mockRegistry.RegisterModel(modelName, "2.0.0", new ModelMetadata());
            _mockLoader.LoadModel(modelName, "1.0.0");
            _mockLoader.LoadModel(modelName, "2.0.0");

            _router.SetDefaultVersion(modelName, "1.0.0");

            var context = new RoutingContext();
            const int iterations = 1000;

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            // Act
            for (int i = 0; i < iterations; i++)
            {
                _router.GetModel(modelName, context);
            }

            stopwatch.Stop();
            var avgTimePerRoute = stopwatch.Elapsed.TotalMilliseconds / iterations;

            // Assert
            Assert.True(avgTimePerRoute < 1.0,
                $"Average routing time {avgTimePerRoute}ms exceeds target of 1.0ms");
        }

        [Fact]
        public void Performance_RoutingWithVersion_Under1ms()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            _mockRegistry.RegisterModel(modelName, version, new ModelMetadata());
            _mockLoader.LoadModel(modelName, version);

            const int iterations = 1000;

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            // Act
            for (int i = 0; i < iterations; i++)
            {
                _router.GetModel(modelName, version);
            }

            stopwatch.Stop();
            var avgTimePerRoute = stopwatch.Elapsed.TotalMilliseconds / iterations;

            // Assert
            Assert.True(avgTimePerRoute < 1.0,
                $"Average routing time {avgTimePerRoute}ms exceeds target of 1.0ms");
        }

        [Fact]
        public async Task ConcurrentRouting_ThreadSafe()
        {
            // Arrange
            const string modelName = "test-model";
            _mockRegistry.RegisterModel(modelName, "1.0.0", new ModelMetadata());
            _mockRegistry.RegisterModel(modelName, "2.0.0", new ModelMetadata());
            _mockLoader.LoadModel(modelName, "1.0.0");
            _mockLoader.LoadModel(modelName, "2.0.0");

            _router.SetDefaultVersion(modelName, "1.0.0");

            const int threadCount = 100;

            var tasks = Enumerable.Range(0, threadCount).Select(i =>
                Task.Run(() =>
                {
                    var context = new RoutingContext();
                    _router.GetModel(modelName, context);
                })
            );

            // Act
            await Task.WhenAll(tasks);

            // Assert - No exceptions thrown, test passes
        }

        [Fact]
        public void RoutingContext_WithHeaders_DoesNotAffectRouting()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            _mockRegistry.RegisterModel(modelName, version, new ModelMetadata());
            var expectedModel = _mockLoader.LoadModel(modelName, version);

            _router.SetDefaultVersion(modelName, version);

            var context = new RoutingContext
            {
                Headers = new Dictionary<string, string>
                {
                    { "X-Feature", "enabled" },
                    { "X-User-Id", "12345" }
                },
                UserId = "user123",
                ExperimentId = "experiment-abc"
            };

            // Act
            var result = _router.GetModel(modelName, context);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(version, result.Version);
            Assert.Same(expectedModel, result);
        }

        // Mock implementations

        private class MockModelRegistry : IModelRegistry
        {
            private readonly Dictionary<string, Dictionary<string, ModelMetadata>> _models =
                new(StringComparer.OrdinalIgnoreCase);

            public void RegisterModel(string name, string version, ModelMetadata metadata)
            {
                if (!_models.ContainsKey(name))
                {
                    _models[name] = new Dictionary<string, ModelMetadata>(StringComparer.OrdinalIgnoreCase);
                }
                _models[name][version] = metadata;
            }

            public void UnregisterModel(string name, string version)
            {
                if (_models.ContainsKey(name))
                {
                    _models[name].Remove(version);
                }
            }

            public bool HasVersion(string name, string version)
            {
                return _models.ContainsKey(name) && _models[name].ContainsKey(version);
            }

            public IEnumerable<string> GetVersions(string name)
            {
                return _models.ContainsKey(name) ? _models[name].Keys : Enumerable.Empty<string>();
            }

            public ModelMetadata GetMetadata(string name, string version)
            {
                if (HasVersion(name, version))
                {
                    return _models[name][version];
                }
                throw new KeyNotFoundException($"Model '{name}' version '{version}' not found");
            }

            public IEnumerable<string> GetAllModelNames()
            {
                return _models.Keys;
            }
        }

        private class MockModelLoader : IModelLoader
        {
            private readonly Dictionary<(string, string), IModel> _loadedModels =
                new();
            private int _instanceCount = 0;

            public IModel Load(string modelPath, string version)
            {
                var modelName = System.IO.Path.GetFileName(modelPath);
                var model = new MockModel(modelName, version, _instanceCount++);
                _loadedModels[(modelName, version)] = model;
                return model;
            }

            public Task<IModel> LoadAsync(string modelPath, string version, CancellationToken ct = default)
            {
                return Task.FromResult(Load(modelPath, version));
            }

            public void Unload(IModel model)
            {
                _loadedModels.Remove((model.Name, model.Version));
            }

            public bool IsLoaded(string name, string version)
            {
                return _loadedModels.ContainsKey((name, version));
            }

            public IEnumerable<IModel> GetLoadedModels()
            {
                return _loadedModels.Values;
            }

            public IModel LoadModel(string name, string version)
            {
                if (!_loadedModels.ContainsKey((name, version)))
                {
                    var model = new MockModel(name, version, _instanceCount++);
                    _loadedModels[(name, version)] = model;
                }
                return _loadedModels[(name, version)];
            }

            public void ClearAllModels()
            {
                _loadedModels.Clear();
            }

            private class MockModel : IModel
            {
                private int _instanceId;
                private bool _disposed = false;

                public MockModel(string name, string version, int instanceId)
                {
                    Name = name;
                    Version = version;
                    LoadTime = DateTime.UtcNow;
                    _instanceId = instanceId;
                    IsActive = true;
                }

                public string Name { get; }
                public string Version { get; }
                public DateTime LoadTime { get; }
                public bool IsActive { get; set; }

                public Task<InferenceResult> InferAsync(InferenceInput input)
                {
                    return Task.FromResult(new InferenceResult(new
                    {
                        InstanceId = _instanceId,
                        Name = Name,
                        Version = Version
                    })
                    {
                        Success = true,
                        InferenceTimeMs = 1
                    });
                }

                public void Dispose()
                {
                    if (!_disposed)
                    {
                        _disposed = true;
                        IsActive = false;
                    }
                }
            }
        }

        private class ConsoleLogger : ILogger<VersionRouter>
        {
            public IDisposable? BeginScope<TState>(TState state) => null;

            public bool IsEnabled(LogLevel logLevel) => true;

            public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
            {
                // Console output for debugging
                // Uncomment to enable logging in tests
                // Console.WriteLine($"[{logLevel}] {formatter(state, exception)}");
            }
        }
    }
}
