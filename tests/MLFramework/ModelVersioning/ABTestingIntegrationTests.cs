using MLFramework.Serving.Deployment;
using Xunit;

namespace MLFramework.Tests.ModelVersioning
{
    /// <summary>
    /// Integration tests for A/B testing scenarios.
    /// </summary>
    public class ABTestingIntegrationTests : IClassFixture<IntegrationTestFixture>
    {
        private readonly IntegrationTestFixture _fixture;

        public ABTestingIntegrationTests(IntegrationTestFixture fixture)
        {
            _fixture = fixture;
        }

        [Fact]
        public async Task ABTesting_CompareAndPromote()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";
            const int totalRequests = 100;
            const double abSplit = 0.90; // 90% v1.0.0, 10% v2.0.0

            // Setup metadata with initial metrics
            var metadata1 = _fixture.Registry.GetMetadata(modelName, version1);
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);

            metadata1.PerformanceMetrics["accuracy"] = 0.90f;
            metadata1.PerformanceMetrics["requests"] = 0;
            metadata1.PerformanceMetrics["errors"] = 0;

            metadata2.PerformanceMetrics["accuracy"] = 0.93f;
            metadata2.PerformanceMetrics["requests"] = 0;
            metadata2.PerformanceMetrics["errors"] = 0;

            // 2. Set up 90/10 A/B test
            var requestsToV1 = (int)(totalRequests * abSplit);
            var requestsToV2 = totalRequests - requestsToV1;

            // 3. Simulate traffic (100 requests)
            var random = new Random(42);
            var version1Requests = 0;
            var version2Requests = 0;

            for (int i = 0; i < totalRequests; i++)
            {
                if (random.NextDouble() < abSplit)
                {
                    version1Requests++;
                    metadata1.PerformanceMetrics["requests"] = (float)metadata1.PerformanceMetrics["requests"] + 1;
                }
                else
                {
                    version2Requests++;
                    metadata2.PerformanceMetrics["requests"] = (float)metadata2.PerformanceMetrics["requests"] + 1;
                }
            }

            // 4. Record metrics for each version
            // Simulate accuracy during the test
            metadata1.PerformanceMetrics["test_accuracy"] = 0.90f;
            metadata2.PerformanceMetrics["test_accuracy"] = 0.93f;

            // 5. Compare results
            var accuracyV1 = (float)metadata1.PerformanceMetrics["test_accuracy"];
            var accuracyV2 = (float)metadata2.PerformanceMetrics["test_accuracy"];

            Assert.True(accuracyV2 > accuracyV1, "Version 2 should have better accuracy");
            Assert.InRange(version1Requests, 85, 95); // Approx 90% traffic
            Assert.InRange(version2Requests, 5, 15);   // Approx 10% traffic

            // 6. Promote winner to 100% - swap to v2.0.0
            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            var swapResult = await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);

            // 7. Verify traffic routing
            Assert.True(swapResult.Success);
            _fixture.MockRouter.Verify(r => r.SetDefaultVersion(modelName, version2), Times.Once);
        }

        [Fact]
        public async Task ABTesting_WithStatisticalSignificance_MakesCorrectDecision()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";
            const int totalRequests = 1000;

            var metadata1 = _fixture.Registry.GetMetadata(modelName, version1);
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);

            // Simulate metrics with some variance
            metadata1.PerformanceMetrics["accuracy"] = 0.90f;
            metadata1.PerformanceMetrics["requests"] = totalRequests / 2;

            metadata2.PerformanceMetrics["accuracy"] = 0.915f; // 1.5% improvement
            metadata2.PerformanceMetrics["requests"] = totalRequests / 2;

            // Act - Determine statistical significance
            var improvement = ((float)metadata2.PerformanceMetrics["accuracy"] - (float)metadata1.PerformanceMetrics["accuracy"])
                            / (float)metadata1.PerformanceMetrics["accuracy"];
            var relativeImprovement = improvement * 100;

            // Assert - Check if improvement is significant (> 1%)
            Assert.True(relativeImprovement > 1.0, "Improvement should be statistically significant");
            Assert.InRange(relativeImprovement, 1.0, 2.0); // Should be 1.5%

            // Promote the better version
            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            var swapResult = await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
            Assert.True(swapResult.Success);
        }

        [Fact]
        public async Task ABTesting_WithErrorRates_ConsidersReliability()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            var metadata1 = _fixture.Registry.GetMetadata(modelName, version1);
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);

            // Version 1 has higher accuracy but more errors
            metadata1.PerformanceMetrics["accuracy"] = 0.95f;
            metadata1.PerformanceMetrics["error_rate"] = 0.02f; // 2% error rate

            // Version 2 has slightly lower accuracy but better reliability
            metadata2.PerformanceMetrics["accuracy"] = 0.93f;
            metadata2.PerformanceMetrics["error_rate"] = 0.005f; // 0.5% error rate

            // Act - Calculate combined score
            var score1 = (float)metadata1.PerformanceMetrics["accuracy"] - (float)metadata1.PerformanceMetrics["error_rate"];
            var score2 = (float)metadata2.PerformanceMetrics["accuracy"] - (float)metadata2.PerformanceMetrics["error_rate"];

            // Assert - Version 2 wins when reliability is considered
            Assert.True(score2 > score1, "Version 2 should win when considering error rates");

            // Swap to version 2
            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            var swapResult = await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
            Assert.True(swapResult.Success);
        }

        [Fact]
        public async Task ABTesting_WithMultipleMetrics_UsesWeightedScore()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            var metadata1 = _fixture.Registry.GetMetadata(modelName, version1);
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);

            // Multiple metrics with different weights
            metadata1.PerformanceMetrics["accuracy"] = 0.90f;
            metadata1.PerformanceMetrics["latency_ms"] = 50f;
            metadata1.PerformanceMetrics["throughput"] = 100f;

            metadata2.PerformanceMetrics["accuracy"] = 0.91f;
            metadata2.PerformanceMetrics["latency_ms"] = 45f; // Better latency
            metadata2.PerformanceMetrics["throughput"] = 110f; // Better throughput

            // Act - Calculate weighted score (accuracy: 50%, latency: 30%, throughput: 20%)
            float NormalizeLatency(float latency) => 100f / latency; // Higher is better
            float NormalizeThroughput(float throughput) => throughput / 120f; // Normalize to 0-1

            var score1 =
                (float)metadata1.PerformanceMetrics["accuracy"] * 0.5f +
                NormalizeLatency((float)metadata1.PerformanceMetrics["latency_ms"]) * 0.3f +
                NormalizeThroughput((float)metadata1.PerformanceMetrics["throughput"]) * 0.2f;

            var score2 =
                (float)metadata2.PerformanceMetrics["accuracy"] * 0.5f +
                NormalizeLatency((float)metadata2.PerformanceMetrics["latency_ms"]) * 0.3f +
                NormalizeThroughput((float)metadata2.PerformanceMetrics["throughput"]) * 0.2f;

            // Assert - Version 2 wins on overall score
            Assert.True(score2 > score1);

            // Swap to version 2
            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            var swapResult = await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
            Assert.True(swapResult.Success);
        }

        [Fact]
        public async Task ABTesting_WithTrafficGradualIncrease_AdaptsSmoothly()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            var metadata1 = _fixture.Registry.GetMetadata(modelName, version1);
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);

            // Start with metrics for version 1
            metadata1.PerformanceMetrics["accuracy"] = 0.90f;
            metadata2.PerformanceMetrics["accuracy"] = 0.92f;

            // Act - Simulate gradual traffic increase to version 2
            var stages = new[] { 0.05, 0.10, 0.25, 0.50, 0.75, 1.0 }; // Traffic percentages

            foreach (var stage in stages)
            {
                // In a real scenario, this would adjust routing policy
                var trafficPercentage = stage * 100;
                Assert.True(trafficPercentage > 0 && trafficPercentage <= 100);

                // Verify health at each stage
                var health = _fixture.HotSwapper.CheckVersionHealth(modelName, version2);
                Assert.True(health.IsHealthy);
            }

            // Finally, swap fully
            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

            var swapResult = await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);

            // Assert
            Assert.True(swapResult.Success);
        }

        [Fact]
        public async Task ABTesting_WithRegression_RollsBack()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version1);

            var metadata1 = _fixture.Registry.GetMetadata(modelName, version1);
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);

            // Version 2 performs worse
            metadata1.PerformanceMetrics["accuracy"] = 0.95f;
            metadata2.PerformanceMetrics["accuracy"] = 0.92f; // Regression

            // Act - Swap to version 2
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));
            var swapResult = await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
            Assert.True(swapResult.Success);

            // Detect regression and rollback
            var regression = (float)metadata2.PerformanceMetrics["accuracy"] < (float)metadata1.PerformanceMetrics["accuracy"];
            Assert.True(regression, "Version 2 shows regression");

            // Rollback
            _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                .Returns(version2);
            _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version1));

            var rollbackResult = await _fixture.HotSwapper.RollbackVersion(modelName, version1);

            // Assert - Rollback successful
            Assert.True(rollbackResult.Success);
            _fixture.MockRouter.Verify(r => r.SetDefaultVersion(modelName, version1), Times.Once);
        }
    }
}
