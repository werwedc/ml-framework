using MLFramework.Serving.Deployment;
using Xunit;

namespace MLFramework.Tests.ModelVersioning
{
    /// <summary>
    /// Integration tests for memory management scenarios.
    /// </summary>
    public class MemoryManagementTests : IClassFixture<IntegrationTestFixture>
    {
        private readonly IntegrationTestFixture _fixture;

        public MemoryManagementTests(IntegrationTestFixture fixture)
        {
            _fixture = fixture;
        }

        [Fact]
        public void Memory_ManageMultipleVersions()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";
            const string version3 = "v1.1.0";

            // Setup memory usage for each version
            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version1))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version1,
                    IsLoaded = true,
                    MemoryUsageBytes = 2L * 1024 * 1024 * 1024, // 2 GB
                    LoadTime = DateTime.UtcNow,
                    RequestCount = 1000
                });

            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version2))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version2,
                    IsLoaded = true,
                    MemoryUsageBytes = 2.5L * 1024 * 1024 * 1024, // 2.5 GB
                    LoadTime = DateTime.UtcNow,
                    RequestCount = 500
                });

            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version3))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version3,
                    IsLoaded = true,
                    MemoryUsageBytes = 1.8L * 1024 * 1024 * 1024, // 1.8 GB
                    LoadTime = DateTime.UtcNow,
                    RequestCount = 100
                });

            // 1. Load multiple versions
            Assert.True(_fixture.VersionManager.IsVersionLoaded(modelName, version1));
            Assert.True(_fixture.VersionManager.IsVersionLoaded(modelName, version2));
            Assert.True(_fixture.VersionManager.IsVersionLoaded(modelName, version3));

            // 2. Verify memory tracking
            var loadInfo1 = _fixture.VersionManager.GetLoadInfo(modelName, version1);
            var loadInfo2 = _fixture.VersionManager.GetLoadInfo(modelName, version2);
            var loadInfo3 = _fixture.VersionManager.GetLoadInfo(modelName, version3);

            var totalMemory = loadInfo1.MemoryUsageBytes +
                             loadInfo2.MemoryUsageBytes +
                             loadInfo3.MemoryUsageBytes;

            var expectedTotal = 6.3 * 1024 * 1024 * 1024; // 6.3 GB
            Assert.Equal(expectedTotal, totalMemory);

            // 3. Unload versions
            _fixture.VersionManager.UnloadVersion(modelName, version2);
            _fixture.MockVersionManager.Setup(m => m.IsVersionLoaded(modelName, version2))
                .Returns(false);

            // 4. Verify memory freed
            var memoryAfterUnload = loadInfo1.MemoryUsageBytes + loadInfo3.MemoryUsageBytes;
            var expectedAfter = 3.8 * 1024 * 1024 * 1024; // 3.8 GB
            Assert.Equal(expectedAfter, memoryAfterUnload);

            // 5. Test memory limits
            var memoryLimit = 4L * 1024 * 1024 * 1024; // 4 GB limit
            var withinLimit = memoryAfterUnload <= memoryLimit;

            Assert.True(withinLimit,
                $"Memory usage {memoryAfterUnload / (1024 * 1024 * 1024)} GB exceeds limit {memoryLimit / (1024 * 1024 * 1024)} GB");
        }

        [Fact]
        public void Memory_WithLargeVersion_DetectsOversized()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            // Version 1 is normal
            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version1))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version1,
                    IsLoaded = true,
                    MemoryUsageBytes = 2L * 1024 * 1024 * 1024, // 2 GB
                    LoadTime = DateTime.UtcNow
                });

            // Version 2 is too large
            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version2))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version2,
                    IsLoaded = true,
                    MemoryUsageBytes = 8L * 1024 * 1024 * 1024, // 8 GB - too large
                    LoadTime = DateTime.UtcNow
                });

            // Act - Check health
            var health1 = _fixture.HotSwapper.CheckVersionHealth(modelName, version1);
            var health2 = _fixture.HotSwapper.CheckVersionHealth(modelName, version2);

            // Assert - Version 2 should be flagged as unhealthy
            Assert.True(health1.IsHealthy);
            Assert.False(health2.IsHealthy);
            Assert.Contains("memory", health2.Message.ToLower());
        }

        [Fact]
        public void Memory_WithMemoryPressure_UnloadsLeastUsed()
        {
            // Arrange
            const string modelName = "test-model";
            var versions = new[]
            {
                "v1.0.0", // 1000 requests
                "v1.1.0", // 500 requests
                "v1.2.0", // 100 requests (least used)
                "v2.0.0"  // 50 requests (least used)
            };

            var requestCounts = new Dictionary<string, int>
            {
                ["v1.0.0"] = 1000,
                ["v1.1.0"] = 500,
                ["v1.2.0"] = 100,
                ["v2.0.0"] = 50
            };

            // Setup load info with request counts
            foreach (var version in versions)
            {
                _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version))
                    .Returns(new VersionLoadInfo
                    {
                        ModelId = modelName,
                        Version = version,
                        IsLoaded = true,
                        MemoryUsageBytes = 2L * 1024 * 1024 * 1024,
                        LoadTime = DateTime.UtcNow,
                        RequestCount = requestCounts[version]
                    });
            }

            // Act - Simulate memory pressure by unloading least used version
            var allLoadInfo = versions.Select(v => _fixture.VersionManager.GetLoadInfo(modelName, v)).ToList();
            var leastUsed = allLoadInfo.OrderBy(info => info.RequestCount).First();

            _fixture.VersionManager.UnloadVersion(modelName, leastUsed.Version);
            _fixture.MockVersionManager.Setup(m => m.IsVersionLoaded(modelName, leastUsed.Version))
                .Returns(false);

            // Assert - Least used version should be unloaded
            Assert.False(_fixture.VersionManager.IsVersionLoaded(modelName, leastUsed.Version));
            Assert.Equal("v2.0.0", leastUsed.Version); // Should be the least used

            // Verify other versions are still loaded
            Assert.True(_fixture.VersionManager.IsVersionLoaded(modelName, "v1.0.0"));
            Assert.True(_fixture.VersionManager.IsVersionLoaded(modelName, "v1.1.0"));
            Assert.True(_fixture.VersionManager.IsVersionLoaded(modelName, "v1.2.0"));
        }

        [Fact]
        public async Task Memory_WithSwap_ManagesEfficiently()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            // Setup load info
            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version1))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version1,
                    IsLoaded = true,
                    MemoryUsageBytes = 2L * 1024 * 1024 * 1024,
                    LoadTime = DateTime.UtcNow,
                    RequestCount = 1000
                });

            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version2))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version2,
                    IsLoaded = true,
                    MemoryUsageBytes = 2L * 1024 * 1024 * 1024,
                    LoadTime = DateTime.UtcNow,
                    RequestCount = 0
                });

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            // Get initial memory
            var initialLoadInfo1 = _fixture.VersionManager.GetLoadInfo(modelName, version1);
            var initialLoadInfo2 = _fixture.VersionManager.GetLoadInfo(modelName, version2);
            var initialMemory = initialLoadInfo1.MemoryUsageBytes + initialLoadInfo2.MemoryUsageBytes;

            // Act - Perform swap
            var swapResult = await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
            Assert.True(swapResult.Success);

            // Get memory after swap
            var finalLoadInfo1 = _fixture.VersionManager.GetLoadInfo(modelName, version1);
            var finalLoadInfo2 = _fixture.VersionManager.GetLoadInfo(modelName, version2);
            var finalMemory = finalLoadInfo1.MemoryUsageBytes + finalLoadInfo2.MemoryUsageBytes;

            // Assert - Memory usage should be stable
            Assert.Equal(initialMemory, finalMemory);
        }

        [Fact]
        public void Memory_WithMemoryLimit_PreventsOversizedLoad()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            const long memoryLimit = 4L * 1024 * 1024 * 1024; // 4 GB limit

            // Version 1 uses 2 GB
            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version1))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version1,
                    IsLoaded = true,
                    MemoryUsageBytes = 2L * 1024 * 1024 * 1024,
                    LoadTime = DateTime.UtcNow
                });

            // Version 2 would use 3 GB (would exceed limit)
            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version2))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version2,
                    IsLoaded = true,
                    MemoryUsageBytes = 3L * 1024 * 1024 * 1024,
                    LoadTime = DateTime.UtcNow
                });

            // Act - Check if version 2 would exceed limit
            var loadInfo1 = _fixture.VersionManager.GetLoadInfo(modelName, version1);
            var loadInfo2 = _fixture.VersionManager.GetLoadInfo(modelName, version2);
            var totalMemory = loadInfo1.MemoryUsageBytes + loadInfo2.MemoryUsageBytes;

            // Assert - Should detect memory limit exceeded
            Assert.True(totalMemory > memoryLimit);

            var healthCheck = _fixture.HotSwapper.CheckVersionHealth(modelName, version2);
            Assert.False(healthCheck.IsHealthy);
        }

        [Fact]
        public void Memory_TrackingAcrossVersions_Accurate()
        {
            // Arrange
            const string modelName = "test-model";
            var versions = new[] { "v1.0.0", "v1.1.0", "v2.0.0", "v2.1.0" };

            var memoryUsages = new Dictionary<string, long>
            {
                ["v1.0.0"] = 2L * 1024 * 1024 * 1024,  // 2 GB
                ["v1.1.0"] = 2.1L * 1024 * 1024 * 1024, // 2.1 GB
                ["v2.0.0"] = 1.8L * 1024 * 1024 * 1024, // 1.8 GB
                ["v2.1.0"] = 1.9L * 1024 * 1024 * 1024  // 1.9 GB
            };

            // Setup load info
            foreach (var version in versions)
            {
                _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version))
                    .Returns(new VersionLoadInfo
                    {
                        ModelId = modelName,
                        Version = version,
                        IsLoaded = true,
                        MemoryUsageBytes = memoryUsages[version],
                        LoadTime = DateTime.UtcNow
                    });
            }

            // Act - Calculate total memory
            var totalMemory = versions
                .Select(v => _fixture.VersionManager.GetLoadInfo(modelName, v).MemoryUsageBytes)
                .Sum();

            var expectedTotal = 7.8 * 1024 * 1024 * 1024; // 7.8 GB

            // Assert
            Assert.Equal(expectedTotal, totalMemory);

            // Verify individual tracking
            foreach (var version in versions)
            {
                var loadInfo = _fixture.VersionManager.GetLoadInfo(modelName, version);
                Assert.Equal(memoryUsages[version], loadInfo.MemoryUsageBytes);
            }
        }

        [Fact]
        public async Task Memory_WithConcurrentOperations_HandlesCorrectly()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version1))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version1,
                    IsLoaded = true,
                    MemoryUsageBytes = 2L * 1024 * 1024 * 1024,
                    LoadTime = DateTime.UtcNow
                });

            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version2))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version2,
                    IsLoaded = true,
                    MemoryUsageBytes = 2L * 1024 * 1024 * 1024,
                    LoadTime = DateTime.UtcNow
                });

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            // Act - Simulate concurrent operations
            var tasks = new List<Task>();

            // Task 1: Load memory info
            tasks.Add(Task.Run(() =>
            {
                var loadInfo = _fixture.VersionManager.GetLoadInfo(modelName, version1);
                Assert.NotNull(loadInfo);
            }));

            // Task 2: Load memory info
            tasks.Add(Task.Run(() =>
            {
                var loadInfo = _fixture.VersionManager.GetLoadInfo(modelName, version2);
                Assert.NotNull(loadInfo);
            }));

            // Task 3: Perform swap
            tasks.Add(_fixture.HotSwapper.SwapVersion(modelName, version1, version2));

            await Task.WhenAll(tasks);

            // Assert - All operations should complete successfully
            Assert.True(tasks.All(t => t.IsCompletedSuccessfully));
        }

        [Fact]
        public void Memory_WithMemoryTrackingAccuracy_PreciseMeasurements()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";

            // Exact memory size
            const long exactMemory = 2L * 1024 * 1024 * 1024 + 512L * 1024 * 1024; // 2.5 GB exactly

            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version1))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version1,
                    IsLoaded = true,
                    MemoryUsageBytes = exactMemory,
                    LoadTime = DateTime.UtcNow
                });

            // Act - Get memory info
            var loadInfo = _fixture.VersionManager.GetLoadInfo(modelName, version1);

            // Assert - Verify exact measurement
            Assert.Equal(exactMemory, loadInfo.MemoryUsageBytes);

            // Convert to different units for verification
            var memoryInMB = exactMemory / (1024 * 1024);
            var expectedMB = 2560; // 2.5 * 1024

            Assert.Equal(expectedMB, memoryInMB);
        }
    }
}
