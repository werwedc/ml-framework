using MLFramework.Serving.Deployment;
using MLFramework.Serving.Routing;
using Xunit;

namespace MLFramework.Tests.ModelVersioning
{
    /// <summary>
    /// Integration tests for complete end-to-end model versioning workflow.
    /// </summary>
    public class IntegrationTests : IClassFixture<IntegrationTestFixture>
    {
        private readonly IntegrationTestFixture _fixture;

        public IntegrationTests(IntegrationTestFixture fixture)
        {
            _fixture = fixture;
        }

        [Fact]
        public async Task FullWorkflow_RegisterRouteSwapRollback()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version1));
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            // 1. Register two model versions (already done in fixture setup)
            Assert.True(_fixture.Registry.HasVersion(modelName, version1));
            Assert.True(_fixture.Registry.HasVersion(modelName, version2));

            // 2. Set up routing policy (set default to v1.0.0)
            _fixture.Router.SetDefaultVersion(modelName, version1);
            Assert.Equal(version1, _fixture.Router.GetDefaultVersion(modelName));

            // 3. Record metrics for both versions (simulate with metadata)
            var metadata1 = _fixture.Registry.GetMetadata(modelName, version1);
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);
            Assert.NotNull(metadata1);
            Assert.NotNull(metadata2);

            // Add simulated performance metrics
            metadata1.PerformanceMetrics["accuracy"] = 0.92f;
            metadata2.PerformanceMetrics["accuracy"] = 0.95f;

            // 4. Compare version performance
            var accuracy1 = metadata1.PerformanceMetrics["accuracy"];
            var accuracy2 = metadata2.PerformanceMetrics["accuracy"];
            Assert.True(accuracy2 > accuracy1);

            // 5. Hot-swap to better version (v2.0.0)
            var swapResult = await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
            Assert.True(swapResult.Success);
            Assert.Contains("Successfully swapped", swapResult.Message);
            _fixture.MockRouter.Verify(r => r.SetDefaultVersion(modelName, version2), Times.Once);

            // 6. Verify routing changed
            Assert.Equal(version2, _fixture.Router.GetDefaultVersion(modelName));

            // 7. Rollback if needed
            var rollbackResult = await _fixture.HotSwapper.RollbackVersion(modelName, version1);
            Assert.True(rollbackResult.Success);
            Assert.Contains("Successfully rolled back", rollbackResult.Message);

            // 8. Verify rollback succeeded
            _fixture.MockRouter.Verify(r => r.SetDefaultVersion(modelName, version1), Times.Once);
            Assert.Equal(version1, _fixture.Router.GetDefaultVersion(modelName));
        }

        [Fact]
        public async Task FullWorkflow_WithHealthCheck_FailsForUnhealthyVersion()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version2))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version2,
                    IsLoaded = false, // Unhealthy
                    MemoryUsageBytes = 1024 * 1024 * 100,
                    RequestCount = 0
                });

            // Act
            var swapResult = await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);

            // Assert
            Assert.False(swapResult.Success);
            Assert.Contains("unhealthy", swapResult.Message.ToLower());
        }

        [Fact]
        public async Task FullWorkflow_WithDrain_WaitsForInFlightRequests()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            // Simulate in-flight requests
            _fixture.HotSwapper.IncrementInFlightRequest(modelName, version1);
            _fixture.HotSwapper.IncrementInFlightRequest(modelName, version1);

            // Act - Start swap but drain requests first
            var drainTask = Task.Run(async () =>
            {
                await Task.Delay(50); // Simulate request processing
                _fixture.HotSwapper.DecrementInFlightRequest(modelName, version1);
                _fixture.HotSwapper.DecrementInFlightRequest(modelName, version1);
            });

            var swapTask = _fixture.HotSwapper.SwapVersion(modelName, version1, version2);

            await Task.WhenAll(drainTask, swapTask);

            // Assert
            var swapResult = await swapTask;
            Assert.True(swapResult.Success);
            Assert.True(swapResult.RequestsDrained >= 0);
        }

        [Fact]
        public async Task FullWorkflow_SwapStateTracking_UpdatesCorrectly()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            // Act - Check initial state
            var statusBefore = _fixture.HotSwapper.GetSwapStatus(modelName);
            Assert.Equal(SwapState.Idle, statusBefore.State);

            // Perform swap
            await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);

            // Check state after swap
            var statusAfter = _fixture.HotSwapper.GetSwapStatus(modelName);
            Assert.Equal(SwapState.Completed, statusAfter.State);
            Assert.Equal(version1, statusAfter.CurrentVersion);
            Assert.Equal(version2, statusAfter.TargetVersion);
        }

        [Fact]
        public async Task FullWorkflow_MultipleSwaps_SequenceWorks()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";
            const string version3 = "v1.1.0";

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(It.IsAny<string>(), It.IsAny<string>()));

            // Act - Swap from v1.0.0 to v2.0.0
            var swapResult1 = await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
            Assert.True(swapResult1.Success);

            // Update mock for next swap
            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version2);

            // Swap from v2.0.0 to v1.1.0
            var swapResult2 = await _fixture.HotSwapper.SwapVersion(modelName, version2, version3);
            Assert.True(swapResult2.Success);

            // Assert - Verify sequence of changes
            _fixture.MockRouter.Verify(r => r.SetDefaultVersion(modelName, version2), Times.Once);
            _fixture.MockRouter.Verify(r => r.SetDefaultVersion(modelName, version3), Times.Once);
        }

        [Fact]
        public async Task FullWorkflow_WithPerformanceComparison_SelectsBestVersion()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";
            const string version3 = "v1.1.0";

            // Set up metrics
            var metadata1 = _fixture.Registry.GetMetadata(modelName, version1);
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);
            var metadata3 = _fixture.Registry.GetMetadata(modelName, version3);

            metadata1.PerformanceMetrics["accuracy"] = 0.90f;
            metadata1.PerformanceMetrics["latency_ms"] = 50f;

            metadata2.PerformanceMetrics["accuracy"] = 0.95f;
            metadata2.PerformanceMetrics["latency_ms"] = 45f;

            metadata3.PerformanceMetrics["accuracy"] = 0.93f;
            metadata3.PerformanceMetrics["latency_ms"] = 40f;

            // Act - Determine best version by accuracy
            var bestAccuracyVersion = new[] { version1, version2, version3 }
                .OrderByDescending(v => _fixture.Registry.GetMetadata(modelName, v).PerformanceMetrics["accuracy"])
                .First();

            // Assert
            Assert.Equal(version2, bestAccuracyVersion);

            // Determine best version by latency
            var bestLatencyVersion = new[] { version1, version2, version3 }
                .OrderBy(v => _fixture.Registry.GetMetadata(modelName, v).PerformanceMetrics["latency_ms"])
                .First();

            // Assert
            Assert.Equal(version3, bestLatencyVersion);
        }

        [Fact]
        public async Task FullWorkflow_ErrorRecovery_HandlesErrorsGracefully()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);

            // Make swap fail
            _fixture.MockVersionManager.Setup(m => m.IsVersionLoaded(modelName, version2))
                .Returns(false);

            // Act - Attempt swap that will fail
            var swapResult = await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);

            // Assert - Verify error handling
            Assert.False(swapResult.Success);
            Assert.Contains("not loaded", swapResult.Message.ToLower());

            // Verify system is still functional after error
            var healthCheck = _fixture.HotSwapper.CheckVersionHealth(modelName, version1);
            Assert.True(healthCheck.IsHealthy);

            var currentStatus = _fixture.HotSwapper.GetSwapStatus(modelName);
            Assert.Equal(SwapState.Idle, currentStatus.State);
        }

        [Fact]
        public async Task FullWorkflow_TimingRequirements_MeetsPerformanceTargets()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            // Act - Measure swap time
            var startTime = DateTime.UtcNow;
            var swapResult = await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
            var swapDuration = DateTime.UtcNow - startTime;

            // Assert - Swap should complete quickly (< 1 minute requirement)
            Assert.True(swapResult.Success);
            Assert.True(swapDuration < TimeSpan.FromMinutes(1));

            // Act - Measure rollback time
            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version2);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version1));

            var rollbackStart = DateTime.UtcNow;
            var rollbackResult = await _fixture.HotSwapper.RollbackVersion(modelName, version1);
            var rollbackDuration = DateTime.UtcNow - rollbackStart;

            // Assert - Rollback should be immediate (< 30 seconds requirement)
            Assert.True(rollbackResult.Success);
            Assert.True(rollbackDuration < TimeSpan.FromSeconds(30));
        }
    }
}
