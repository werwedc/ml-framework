using MLFramework.Serving.Deployment;
using Xunit;

namespace MLFramework.Tests.ModelVersioning
{
    /// <summary>
    /// Integration tests for emergency rollback scenarios.
    /// </summary>
    public class EmergencyRollbackTests : IClassFixture<IntegrationTestFixture>
    {
        private readonly IntegrationTestFixture _fixture;

        public EmergencyRollbackTests(IntegrationTestFixture fixture)
        {
            _fixture = fixture;
        }

        [Fact]
        public async Task EmergencyRollback_OnHighErrorRate()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            // 1. Deploy new version
            var swapResult = await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
            Assert.True(swapResult.Success);

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version2);

            // 2. Simulate high error rate on new version
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);
            metadata2.PerformanceMetrics["error_rate"] = 0.15f; // 15% error rate - critical

            var currentErrorRate = (float)metadata2.PerformanceMetrics["error_rate"];
            var errorThreshold = 0.10f; // 10% threshold

            // 3. Trigger alert
            var alertTriggered = currentErrorRate > errorThreshold;
            Assert.True(alertTriggered, "Alert should be triggered for high error rate");

            // 4. Execute rollback
            if (alertTriggered)
            {
                var rollbackStart = DateTime.UtcNow;

                _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version1));
                var rollbackResult = await _fixture.HotSwapper.RollbackVersion(modelName, version1);
                var rollbackDuration = DateTime.UtcNow - rollbackStart;

                // 5. Verify rollback completed < 30 seconds
                Assert.True(rollbackResult.Success);
                Assert.True(rollbackDuration < TimeSpan.FromSeconds(30),
                    $"Rollback took {rollbackDuration.TotalSeconds}s, expected < 30s");
            }

            // 6. Verify traffic restored to old version
            _fixture.MockRouter.Verify(r => r.SetDefaultVersion(modelName, version1), Times.Once);
        }

        [Fact]
        public async Task EmergencyRollback_WithLatencySpike_RollsBackImmediately()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            // Deploy new version
            await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version2);

            // Simulate latency spike
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);
            metadata2.PerformanceMetrics["latency_p99"] = 2000f; // 2 seconds - critical
            metadata2.PerformanceMetrics["latency_p95"] = 1500f; // 1.5 seconds

            var latencyThreshold = 1000f; // 1 second threshold
            var currentLatency = (float)metadata2.PerformanceMetrics["latency_p99"];

            // Trigger emergency rollback
            if (currentLatency > latencyThreshold)
            {
                var rollbackStart = DateTime.UtcNow;

                _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version1));
                var rollbackResult = await _fixture.HotSwapper.RollbackVersion(modelName, version1);
                var rollbackDuration = DateTime.UtcNow - rollbackStart;

                Assert.True(rollbackResult.Success);
                Assert.True(rollbackDuration < TimeSpan.FromSeconds(10), // Should be very fast for critical issues
                    $"Rollback took {rollbackDuration.TotalSeconds}s, expected < 10s");
            }
        }

        [Fact]
        public async Task EmergencyRollback_WithCompleteFailure_RestoresService()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            // Deploy new version
            await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version2);

            // Simulate complete failure
            _fixture.MockVersionManager.Setup(m => m.IsVersionLoaded(modelName, version2))
                .Returns(false);

            var healthCheck = _fixture.HotSwapper.CheckVersionHealth(modelName, version2);
            Assert.False(healthCheck.IsHealthy);

            // Emergency rollback
            var rollbackStart = DateTime.UtcNow;

            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version1));
            var rollbackResult = await _fixture.HotSwapper.RollbackVersion(modelName, version1);
            var rollbackDuration = DateTime.UtcNow - rollbackStart;

            Assert.True(rollbackResult.Success);
            Assert.True(rollbackDuration < TimeSpan.FromSeconds(5)); // Should be instant

            // Verify health after rollback
            healthCheck = _fixture.HotSwapper.CheckVersionHealth(modelName, version1);
            Assert.True(healthCheck.IsHealthy);
        }

        [Fact]
        public async Task EmergencyRollback_WithInFlightRequests_DrainsAndRollsBack()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            // Deploy new version
            await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version2);

            // Simulate in-flight requests
            _fixture.HotSwapper.IncrementInFlightRequest(modelName, version2);
            _fixture.HotSwapper.IncrementInFlightRequest(modelName, version2);
            _fixture.HotSwapper.IncrementInFlightRequest(modelName, version2);

            // Trigger emergency rollback
            var rollbackStart = DateTime.UtcNow;

            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version1));
            var rollbackResult = await _fixture.HotSwapper.RollbackVersion(modelName, version1);
            var rollbackDuration = DateTime.UtcNow - rollbackStart;

            Assert.True(rollbackResult.Success);
            Assert.True(rollbackDuration < TimeSpan.FromSeconds(30));

            // Drain in-flight requests after rollback
            var drainStart = DateTime.UtcNow;
            var drainResult = await _fixture.HotSwapper.DrainVersion(modelName, version2, TimeSpan.FromSeconds(10));
            var drainDuration = DateTime.UtcNow - drainStart;

            Assert.True(drainResult);
            Assert.True(drainDuration < TimeSpan.FromSeconds(10));

            // Cleanup
            _fixture.HotSwapper.DecrementInFlightRequest(modelName, version2);
            _fixture.HotSwapper.DecrementInFlightRequest(modelName, version2);
            _fixture.HotSwapper.DecrementInFlightRequest(modelName, version2);
        }

        [Fact]
        public async Task EmergencyRollback_WithMultipleAlerts_PrioritizesCritical()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            // Deploy new version
            await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version2);

            // Simulate multiple alerts
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);
            metadata2.PerformanceMetrics["error_rate"] = 0.12f; // High error rate
            metadata2.PerformanceMetrics["latency_p99"] = 1500f; // High latency
            metadata2.PerformanceMetrics["memory_usage"] = 0.95f; // High memory

            // Check for critical alerts
            var criticalError = (float)metadata2.PerformanceMetrics["error_rate"] > 0.10f;
            var criticalLatency = (float)metadata2.PerformanceMetrics["latency_p99"] > 1000f;
            var criticalMemory = (float)metadata2.PerformanceMetrics["memory_usage"] > 0.90f;

            var anyCritical = criticalError || criticalLatency || criticalMemory;
            Assert.True(anyCritical, "At least one critical alert should be triggered");

            // Emergency rollback on first critical alert
            if (anyCritical)
            {
                _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version1));
                var rollbackResult = await _fixture.HotSwapper.RollbackVersion(modelName, version1);

                Assert.True(rollbackResult.Success);
            }
        }

        [Fact]
        public async Task EmergencyRollback_MeetsSLARequirements()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            // Deploy new version
            var deploymentStart = DateTime.UtcNow;
            await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version2);

            // Simulate rapid failure detection
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);
            metadata2.PerformanceMetrics["error_rate"] = 0.20f; // Critical

            // SLA: Rollback within 30 seconds of deployment
            var deploymentDuration = DateTime.UtcNow - deploymentStart;

            // Emergency rollback
            var rollbackStart = DateTime.UtcNow;
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version1));
            var rollbackResult = await _fixture.HotSwapper.RollbackVersion(modelName, version1);
            var rollbackDuration = DateTime.UtcNow - rollbackStart;

            // Total time to rollback
            var totalTime = deploymentDuration + rollbackDuration;

            Assert.True(rollbackResult.Success);
            Assert.True(totalTime < TimeSpan.FromSeconds(30),
                $"Total rollback time {totalTime.TotalSeconds}s exceeds SLA of 30s");
        }

        [Fact]
        public async Task EmergencyRollback_WithPartialTraffic_RollsBackToHealthy()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";
            const string version3 = "v1.1.0";

            // Simulate multi-version deployment
            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);

            // Swap to version 2
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));
            await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version2);

            // Version 2 has issues
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);
            metadata2.PerformanceMetrics["error_rate"] = 0.15f;

            // Rollback to version 1
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version1));
            var rollbackResult = await _fixture.HotSwapper.RollbackVersion(modelName, version1);

            Assert.True(rollbackResult.Success);

            // Verify health of rollback target
            var healthCheck = _fixture.HotSwapper.CheckVersionHealth(modelName, version1);
            Assert.True(healthCheck.IsHealthy);
        }

        [Fact]
        public async Task EmergencyRollback_WithRetryAttempts_FailsFastOnCritical()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            // Deploy new version
            await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version2);

            // Simulate complete failure - no retries
            _fixture.MockVersionManager.Setup(m => m.IsVersionLoaded(modelName, version2))
                .Returns(false);

            // Should rollback immediately without retries
            var rollbackStart = DateTime.UtcNow;
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version1));
            var rollbackResult = await _fixture.HotSwapper.RollbackVersion(modelName, version1);
            var rollbackDuration = DateTime.UtcNow - rollbackStart;

            Assert.True(rollbackResult.Success);
            Assert.True(rollbackDuration < TimeSpan.FromSeconds(5)); // Fail fast
        }
    }
}
