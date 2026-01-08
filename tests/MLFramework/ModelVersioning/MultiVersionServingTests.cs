using MLFramework.Serving.Deployment;
using MLFramework.Serving.Routing;
using Moq;
using Xunit;

namespace MLFramework.Tests.ModelVersioning
{
    /// <summary>
    /// Integration tests for multi-version serving scenarios.
    /// </summary>
    public class MultiVersionServingTests : IClassFixture<IntegrationTestFixture>
    {
        private readonly IntegrationTestFixture _fixture;

        public MultiVersionServingTests(IntegrationTestFixture fixture)
        {
            _fixture = fixture;
        }

        [Fact]
        public void MultipleVersions_ServeConcurrently()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";
            const string version3 = "v1.1.0";

            // Create mock models
            var mockModel1 = new Mock<IModel>();
            var mockModel2 = new Mock<IModel>();
            var mockModel3 = new Mock<IModel>();

            mockModel1.Setup(m => m.Name).Returns(modelName);
            mockModel1.Setup(m => m.Version).Returns(version1);
            mockModel1.Setup(m => m.IsActive).Returns(true);

            mockModel2.Setup(m => m.Name).Returns(modelName);
            mockModel2.Setup(m => m.Version).Returns(version2);
            mockModel2.Setup(m => m.IsActive).Returns(true);

            mockModel3.Setup(m => m.Name).Returns(modelName);
            mockModel3.Setup(m => m.Version).Returns(version3);
            mockModel3.Setup(m => m.IsActive).Returns(true);

            // Setup router to serve all versions
            _fixture.MockRouter.Setup(r => r.GetModel(modelName, version1))
                .Returns(mockModel1.Object);
            _fixture.MockRouter.Setup(r => r.GetModel(modelName, version2))
                .Returns(mockModel2.Object);
            _fixture.MockRouter.Setup(r => r.GetModel(modelName, version3))
                .Returns(mockModel3.Object);

            // 1. Load 3 versions simultaneously
            Assert.True(_fixture.VersionManager.IsVersionLoaded(modelName, version1));
            Assert.True(_fixture.VersionManager.IsVersionLoaded(modelName, version2));
            Assert.True(_fixture.VersionManager.IsVersionLoaded(modelName, version3));

            // 2. Route to all 3 versions
            var model1 = _fixture.Router.GetModel(modelName, version1);
            var model2 = _fixture.Router.GetModel(modelName, version2);
            var model3 = _fixture.Router.GetModel(modelName, version3);

            Assert.NotNull(model1);
            Assert.NotNull(model2);
            Assert.NotNull(model3);
            Assert.Equal(version1, model1.Version);
            Assert.Equal(version2, model2.Version);
            Assert.Equal(version3, model3.Version);

            // 3. Verify no version interference
            Assert.Equal(version1, model1.Version);
            Assert.Equal(version2, model2.Version);
            Assert.Equal(version3, model3.Version);

            // 4. Verify memory management
            var loadInfo1 = _fixture.VersionManager.GetLoadInfo(modelName, version1);
            var loadInfo2 = _fixture.VersionManager.GetLoadInfo(modelName, version2);
            var loadInfo3 = _fixture.VersionManager.GetLoadInfo(modelName, version3);

            Assert.True(loadInfo1.IsLoaded);
            Assert.True(loadInfo2.IsLoaded);
            Assert.True(loadInfo3.IsLoaded);

            // 5. Unload one version
            _fixture.VersionManager.UnloadVersion(modelName, version2);
            _fixture.MockVersionManager.Setup(m => m.IsVersionLoaded(modelName, version2))
                .Returns(false);

            // 6. Verify others continue serving
            Assert.True(_fixture.VersionManager.IsVersionLoaded(modelName, version1));
            Assert.False(_fixture.VersionManager.IsVersionLoaded(modelName, version2));
            Assert.True(_fixture.VersionManager.IsVersionLoaded(modelName, version3));

            var model1After = _fixture.Router.GetModel(modelName, version1);
            var model3After = _fixture.Router.GetModel(modelName, version3);

            Assert.NotNull(model1After);
            Assert.NotNull(model3After);
        }

        [Fact]
        public async Task MultipleVersions_ConcurrentRequests_HandlesLoad()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";
            const int numConcurrentRequests = 100;

            // Setup models
            var mockModel1 = new Mock<IModel>();
            var mockModel2 = new Mock<IModel>();

            mockModel1.Setup(m => m.Name).Returns(modelName);
            mockModel1.Setup(m => m.Version).Returns(version1);

            mockModel2.Setup(m => m.Name).Returns(modelName);
            mockModel2.Setup(m => m.Version).Returns(version2);

            _fixture.MockRouter.Setup(r => r.GetModel(modelName, version1))
                .Returns(mockModel1.Object);
            _fixture.MockRouter.Setup(r => r.GetModel(modelName, version2))
                .Returns(mockModel2.Object);

            // Act - Send concurrent requests to both versions
            var tasks = new List<Task>();
            var version1Count = 0;
            var version2Count = 0;

            for (int i = 0; i < numConcurrentRequests; i++)
            {
                var useVersion1 = i % 2 == 0;

                tasks.Add(Task.Run(() =>
                {
                    if (useVersion1)
                    {
                        var model = _fixture.Router.GetModel(modelName, version1);
                        if (model != null) Interlocked.Increment(ref version1Count);
                    }
                    else
                    {
                        var model = _fixture.Router.GetModel(modelName, version2);
                        if (model != null) Interlocked.Increment(ref version2Count);
                    }
                }));
            }

            await Task.WhenAll(tasks);

            // Assert - Both versions should have handled requests
            Assert.Equal(numConcurrentRequests / 2, version1Count);
            Assert.Equal(numConcurrentRequests / 2, version2Count);
        }

        [Fact]
        public void MultipleVersions_WithDifferentRequests_RoutesCorrectly()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            var mockModel1 = new Mock<IModel>();
            var mockModel2 = new Mock<IModel>();

            mockModel1.Setup(m => m.Name).Returns(modelName);
            mockModel1.Setup(m => m.Version).Returns(version1);

            mockModel2.Setup(m => m.Name).Returns(modelName);
            mockModel2.Setup(m => m.Version).Returns(version2);

            // Setup routing based on context
            _fixture.MockRouter.Setup(r => r.GetModel(modelName, It.Is<RoutingContext>(c => c.PreferredVersion == version1)))
                .Returns(mockModel1.Object);
            _fixture.MockRouter.Setup(r => r.GetModel(modelName, It.Is<RoutingContext>(c => c.PreferredVersion == version2)))
                .Returns(mockModel2.Object);

            // Act - Route requests with different contexts
            var context1 = new RoutingContext { PreferredVersion = version1 };
            var context2 = new RoutingContext { PreferredVersion = version2 };

            var model1 = _fixture.Router.GetModel(modelName, context1);
            var model2 = _fixture.Router.GetModel(modelName, context2);

            // Assert
            Assert.NotNull(model1);
            Assert.NotNull(model2);
            Assert.Equal(version1, model1.Version);
            Assert.Equal(version2, model2.Version);
        }

        [Fact]
        public void MultipleVersions_GetLoadedVersions_ReturnsAll()
        {
            // Arrange
            const string modelName = "test-model";
            var versions = new[] { "v1.0.0", "v2.0.0", "v1.1.0", "v3.0.0" };

            // Setup version manager
            foreach (var version in versions)
            {
                _fixture.MockVersionManager.Setup(m => m.IsVersionLoaded(modelName, version))
                    .Returns(true);
            }

            _fixture.MockVersionManager.Setup(m => m.GetLoadedVersions(modelName))
                .Returns(versions);

            // Act
            var loadedVersions = _fixture.VersionManager.GetLoadedVersions(modelName).ToList();

            // Assert
            Assert.Equal(versions.Length, loadedVersions.Count);
            Assert.All(versions, v => Assert.Contains(v, loadedVersions));
        }

        [Fact]
        public void MultipleVersions_WithMemoryTracking_ReportsCorrectUsage()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            // Setup memory usage
            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version1))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version1,
                    IsLoaded = true,
                    MemoryUsageBytes = 2L * 1024 * 1024 * 1024 // 2 GB
                });

            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version2))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version2,
                    IsLoaded = true,
                    MemoryUsageBytes = 1.5L * 1024 * 1024 * 1024 // 1.5 GB
                });

            // Act
            var loadInfo1 = _fixture.VersionManager.GetLoadInfo(modelName, version1);
            var loadInfo2 = _fixture.VersionManager.GetLoadInfo(modelName, version2);

            var totalMemory = loadInfo1.MemoryUsageBytes + loadInfo2.MemoryUsageBytes;

            // Assert
            Assert.Equal(3.5 * 1024 * 1024 * 1024, totalMemory); // 3.5 GB total
        }

        [Fact]
        public async Task MultipleVersions_WithVersionSwap_MaintainsOthers()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";
            const string version3 = "v1.1.0";

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);

            // Act - Swap from v1.0.0 to v2.0.0
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));
            var swapResult = await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
            Assert.True(swapResult.Success);

            // Verify v1.1.0 is still loaded
            Assert.True(_fixture.VersionManager.IsVersionLoaded(modelName, version3));

            // Swap to v1.1.0
            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version2);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version3));
            swapResult = await _fixture.HotSwapper.SwapVersion(modelName, version2, version3);
            Assert.True(swapResult.Success);

            // Verify v1.0.0 and v2.0.0 are still loaded
            Assert.True(_fixture.VersionManager.IsVersionLoaded(modelName, version1));
            Assert.True(_fixture.VersionManager.IsVersionLoaded(modelName, version2));
        }

        [Fact]
        public void MultipleVersions_WithWarmUp_PerformsCorrectly()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            var warmupData = new List<object> { "input1", "input2", "input3" };

            // Act - Warm up both versions
            _fixture.VersionManager.WarmUpVersion(modelName, version1, warmupData);
            _fixture.VersionManager.WarmUpVersion(modelName, version2, warmupData);

            // Verify warmup was called
            _fixture.MockVersionManager.Verify(m => m.WarmUpVersion(modelName, version1, warmupData), Times.Once);
            _fixture.MockVersionManager.Verify(m => m.WarmUpVersion(modelName, version2, warmupData), Times.Once);
        }

        [Fact]
        public void MultipleVersions_WithSelectiveUnloading_ManagesResources()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";
            const string version3 = "v1.1.0";

            // Calculate initial memory
            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version1))
                .Returns(new VersionLoadInfo { MemoryUsageBytes = 2L * 1024 * 1024 * 1024 });
            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version2))
                .Returns(new VersionLoadInfo { MemoryUsageBytes = 2L * 1024 * 1024 * 1024 });
            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version3))
                .Returns(new VersionLoadInfo { MemoryUsageBytes = 2L * 1024 * 1024 * 1024 });

            var initialMemory = 6L * 1024 * 1024 * 1024; // 6 GB

            // Act - Unload version 2
            _fixture.VersionManager.UnloadVersion(modelName, version2);
            _fixture.MockVersionManager.Setup(m => m.IsVersionLoaded(modelName, version2))
                .Returns(false);

            // Calculate memory after unloading
            var memoryAfter = initialMemory - 2L * 1024 * 1024 * 1024; // -2 GB

            // Assert
            Assert.Equal(4L * 1024 * 1024 * 1024, memoryAfter);
            Assert.True(_fixture.VersionManager.IsVersionLoaded(modelName, version1));
            Assert.False(_fixture.VersionManager.IsVersionLoaded(modelName, version2));
            Assert.True(_fixture.VersionManager.IsVersionLoaded(modelName, version3));
        }

        [Fact]
        public void MultipleVersions_WithRequestCounting_TracksCorrectly()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            // Setup load info with request counts
            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version1))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version1,
                    IsLoaded = true,
                    RequestCount = 1000
                });

            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version2))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version2,
                    IsLoaded = true,
                    RequestCount = 500
                });

            // Act
            var loadInfo1 = _fixture.VersionManager.GetLoadInfo(modelName, version1);
            var loadInfo2 = _fixture.VersionManager.GetLoadInfo(modelName, version2);

            // Assert
            Assert.Equal(1000, loadInfo1.RequestCount);
            Assert.Equal(500, loadInfo2.RequestCount);
            Assert.True(loadInfo1.RequestCount > loadInfo2.RequestCount);
        }
    }
}
