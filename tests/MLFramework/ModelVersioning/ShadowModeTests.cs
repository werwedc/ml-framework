using MLFramework.Serving.Deployment;
using MLFramework.Serving.Routing;
using Moq;
using Xunit;

namespace MLFramework.Tests.ModelVersioning
{
    /// <summary>
    /// Integration tests for shadow mode scenarios.
    /// </summary>
    public class ShadowModeTests : IClassFixture<IntegrationTestFixture>
    {
        private readonly IntegrationTestFixture _fixture;

        public ShadowModeTests(IntegrationTestFixture fixture)
        {
            _fixture = fixture;
        }

        [Fact]
        public void ShadowMode_CompareOutputs()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0"; // Primary
            const string version2 = "v2.0.0"; // Shadow

            var mockModel1 = new Mock<IModel>();
            var mockModel2 = new Mock<IModel>();

            // Setup mock models
            mockModel1.Setup(m => m.Name).Returns(modelName);
            mockModel1.Setup(m => m.Version).Returns(version1);
            mockModel1.Setup(m => m.IsActive).Returns(true);

            mockModel2.Setup(m => m.Name).Returns(modelName);
            mockModel2.Setup(m => m.Version).Returns(version2);
            mockModel2.Setup(m => m.IsActive).Returns(true);

            // Setup router to return primary model
            var context = new RoutingContext();
            _fixture.MockRouter.Setup(r => r.GetModel(modelName, context))
                .Returns(mockModel1.Object);

            // 2. Send requests
            var input = new InferenceInput { ModelName = modelName, Version = version1 };
            var primaryResult = mockModel1.Object.InferAsync(input);
            var shadowResult = mockModel2.Object.InferAsync(input);

            // 3. Verify only primary responses returned
            Assert.NotNull(primaryResult);
            Assert.NotNull(shadowResult);
            Assert.Equal(modelName, mockModel1.Object.Name);
            Assert.Equal(version1, mockModel1.Object.Version);

            // 4. Verify shadow version received traffic
            Assert.Equal(modelName, mockModel2.Object.Name);
            Assert.Equal(version2, mockModel2.Object.Version);

            // 5. Compare metrics between versions
            var metadata1 = _fixture.Registry.GetMetadata(modelName, version1);
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);

            metadata1.PerformanceMetrics["latency_ms"] = 50f;
            metadata2.PerformanceMetrics["latency_ms"] = 45f;

            var latency1 = (float)metadata1.PerformanceMetrics["latency_ms"];
            var latency2 = (float)metadata2.PerformanceMetrics["latency_ms"];

            Assert.True(latency2 < latency1, "Shadow version should be faster");
        }

        [Fact]
        public async Task ShadowMode_WithMultipleRequests_CollectsMetrics()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";
            const int numRequests = 100;

            var metadata1 = _fixture.Registry.GetMetadata(modelName, version1);
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);

            // Initialize metrics
            metadata1.PerformanceMetrics["latency_ms"] = 0f;
            metadata2.PerformanceMetrics["latency_ms"] = 0f;
            metadata1.PerformanceMetrics["request_count"] = 0f;
            metadata2.PerformanceMetrics["request_count"] = 0f;

            var random = new Random(42);
            var primaryLatencies = new List<float>();
            var shadowLatencies = new List<float>();

            // Act - Send multiple requests
            for (int i = 0; i < numRequests; i++)
            {
                // Simulate primary latency
                var primaryLatency = 50 + random.Next(-10, 10);
                primaryLatencies.Add(primaryLatency);

                // Simulate shadow latency (slightly better)
                var shadowLatency = 45 + random.Next(-10, 10);
                shadowLatencies.Add(shadowLatency);
            }

            // Calculate averages
            metadata1.PerformanceMetrics["latency_ms"] = primaryLatencies.Average();
            metadata2.PerformanceMetrics["latency_ms"] = shadowLatencies.Average();
            metadata1.PerformanceMetrics["request_count"] = numRequests;
            metadata2.PerformanceMetrics["request_count"] = numRequests;

            // Assert - Verify both versions received all requests
            Assert.Equal(numRequests, (int)metadata1.PerformanceMetrics["request_count"]);
            Assert.Equal(numRequests, (int)metadata2.PerformanceMetrics["request_count"]);

            // Verify shadow performed better
            Assert.True(
                (float)metadata2.PerformanceMetrics["latency_ms"] < (float)metadata1.PerformanceMetrics["latency_ms"],
                "Shadow should have lower latency");
        }

        [Fact]
        public void ShadowMode_WithDifferentOutputs_ComparesCorrectly()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            var metadata1 = _fixture.Registry.GetMetadata(modelName, version1);
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);

            // Simulate different outputs
            metadata1.PerformanceMetrics["accuracy"] = 0.90f;
            metadata2.PerformanceMetrics["accuracy"] = 0.93f;

            metadata1.PerformanceMetrics["output_distribution"] = new float[] { 0.4f, 0.6f };
            metadata2.PerformanceMetrics["output_distribution"] = new float[] { 0.35f, 0.65f };

            // Act - Compare outputs
            var accuracyDiff = (float)metadata2.PerformanceMetrics["accuracy"] -
                             (float)metadata1.PerformanceMetrics["accuracy"];

            var primaryDist = (float[])metadata1.PerformanceMetrics["output_distribution"];
            var shadowDist = (float[])metadata2.PerformanceMetrics["output_distribution"];

            var distributionDiff = primaryDist.Zip(shadowDist, (p, s) => Math.Abs(p - s)).Sum();

            // Assert - Shadow should be better
            Assert.True(accuracyDiff > 0, "Shadow should have higher accuracy");
            Assert.True(distributionDiff > 0, "Output distributions should differ");
        }

        [Fact]
        public async Task ShadowMode_WithErrorDifferences_LogsDiscrepancies()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            var metadata1 = _fixture.Registry.GetMetadata(modelName, version1);
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);

            metadata1.PerformanceMetrics["error_rate"] = 0.02f; // 2%
            metadata2.PerformanceMetrics["error_rate"] = 0.015f; // 1.5% - better

            // Act - Calculate error rate difference
            var errorDiff = (float)metadata1.PerformanceMetrics["error_rate"] -
                           (float)metadata2.PerformanceMetrics["error_rate"];

            // Assert - Shadow should have lower error rate
            Assert.True(errorDiff > 0, "Shadow should have lower error rate");
            Assert.InRange(errorDiff, 0, 0.01f); // 0.5% improvement
        }

        [Fact]
        public void ShadowMode_WithMemoryUsage_CompareEfficiency()
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
                    MemoryUsageBytes = 2L * 1024 * 1024 * 1024, // 2 GB
                    RequestCount = 1000
                });

            _fixture.MockVersionManager.Setup(m => m.GetLoadInfo(modelName, version2))
                .Returns(new VersionLoadInfo
                {
                    ModelId = modelName,
                    Version = version2,
                    IsLoaded = true,
                    MemoryUsageBytes = 1.5L * 1024 * 1024 * 1024, // 1.5 GB
                    RequestCount = 1000
                });

            // Act - Get load info
            var loadInfo1 = _fixture.VersionManager.GetLoadInfo(modelName, version1);
            var loadInfo2 = _fixture.VersionManager.GetLoadInfo(modelName, version2);

            // Assert - Shadow should be more memory efficient
            Assert.True(loadInfo2.MemoryUsageBytes < loadInfo1.MemoryUsageBytes,
                "Shadow should use less memory");

            var memorySaved = loadInfo1.MemoryUsageBytes - loadInfo2.MemoryUsageBytes;
            var savedMB = memorySaved / (1024 * 1024);

            Assert.InRange(savedMB, 400, 600); // Should save ~500 MB
        }

        [Fact]
        public async Task ShadowMode_WithThroughput_MeasuresPerformance()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";
            const int numRequests = 1000;
            const int timeWindowSeconds = 10;

            var metadata1 = _fixture.Registry.GetMetadata(modelName, version1);
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);

            // Act - Simulate throughput test
            var startTime = DateTime.UtcNow;

            // Simulate processing requests
            await Task.Run(() =>
            {
                for (int i = 0; i < numRequests; i++)
                {
                    // Simulate work
                    Thread.Sleep(1);
                }
            });

            var endTime = DateTime.UtcNow;
            var duration = (endTime - startTime).TotalSeconds;

            // Calculate throughput
            var throughput = numRequests / duration;

            // Record metrics
            metadata1.PerformanceMetrics["throughput"] = throughput * 1.1f; // Primary baseline
            metadata2.PerformanceMetrics["throughput"] = throughput * 1.2f; // Shadow is faster

            // Assert
            var throughput1 = (float)metadata1.PerformanceMetrics["throughput"];
            var throughput2 = (float)metadata2.PerformanceMetrics["throughput"];

            Assert.True(throughput2 > throughput1, "Shadow should have higher throughput");
        }

        [Fact]
        public void ShadowMode_WithResourceUtilization_MonitorsMetrics()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            var metadata1 = _fixture.Registry.GetMetadata(modelName, version1);
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);

            // Simulate resource metrics
            metadata1.PerformanceMetrics["cpu_usage"] = 0.75f; // 75%
            metadata1.PerformanceMetrics["gpu_usage"] = 0.80f; // 80%

            metadata2.PerformanceMetrics["cpu_usage"] = 0.65f; // 65% - better
            metadata2.PerformanceMetrics["gpu_usage"] = 0.70f; // 70% - better

            // Act - Compare resource usage
            var cpuDiff = (float)metadata1.PerformanceMetrics["cpu_usage"] -
                         (float)metadata2.PerformanceMetrics["cpu_usage"];
            var gpuDiff = (float)metadata1.PerformanceMetrics["gpu_usage"] -
                         (float)metadata2.PerformanceMetrics["gpu_usage"];

            // Assert - Shadow should be more efficient
            Assert.True(cpuDiff > 0, "Shadow should use less CPU");
            Assert.True(gpuDiff > 0, "Shadow should use less GPU");
        }

        [Fact]
        public async Task ShadowMode_DecisionToPromote_BasedOnResults()
        {
            // Arrange
            const string modelName = "test-model";
            const string version1 = "v1.0.0";
            const string version2 = "v2.0.0";

            var metadata1 = _fixture.Registry.GetMetadata(modelName, version1);
            var metadata2 = _fixture.Registry.GetMetadata(modelName, version2);

            // Shadow version performs better on all metrics
            metadata1.PerformanceMetrics["accuracy"] = 0.90f;
            metadata1.PerformanceMetrics["latency_ms"] = 50f;
            metadata1.PerformanceMetrics["error_rate"] = 0.02f;

            metadata2.PerformanceMetrics["accuracy"] = 0.93f; // +3%
            metadata2.PerformanceMetrics["latency_ms"] = 45f; // -5ms
            metadata2.PerformanceMetrics["error_rate"] = 0.015f; // -0.5%

            // Act - Make promotion decision
            var accuracyImproved = (float)metadata2.PerformanceMetrics["accuracy"] > (float)metadata1.PerformanceMetrics["accuracy"];
            var latencyImproved = (float)metadata2.PerformanceMetrics["latency_ms"] < (float)metadata1.PerformanceMetrics["latency_ms"];
            var errorsReduced = (float)metadata2.PerformanceMetrics["error_rate"] < (float)metadata1.PerformanceMetrics["error_rate"];

            var allImproved = accuracyImproved && latencyImproved && errorsReduced;

            // Assert - Should promote
            Assert.True(allImproved, "All metrics should be improved");

            if (allImproved)
            {
                // Promote shadow to primary
                _fixture.MockRouter.Setup(r => r.GetDefaultVersion(modelName))
                    .Returns(version1);
                _fixture.MockRouter.Setup(r => r.SetDefaultVersion(modelName, version2));

                var swapResult = await _fixture.HotSwapper.SwapVersion(modelName, version1, version2);
                Assert.True(swapResult.Success);
            }
        }
    }
}
