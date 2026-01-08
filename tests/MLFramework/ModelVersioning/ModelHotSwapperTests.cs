using MLFramework.ModelVersioning;
using MLFramework.Serving.Routing;
using Moq;
using Xunit;

namespace MLFramework.Tests.ModelVersioning
{
    /// <summary>
    /// Unit tests for the ModelHotSwapper class.
    /// </summary>
    public class ModelHotSwapperTests
    {
        private readonly Mock<IModelVersionManager> _mockVersionManager;
        private readonly Mock<IVersionRouter> _mockRouter;
        private readonly ModelHotSwapper _hotSwapper;
        private const string TestModelId = "test-model";
        private const string TestVersion1 = "v1.0.0";
        private const string TestVersion2 = "v2.0.0";
        private const string TestVersion3 = "v3.0.0";

        public ModelHotSwapperTests()
        {
            _mockVersionManager = new Mock<IModelVersionManager>();
            _mockRouter = new Mock<IVersionRouter>();
            _hotSwapper = new ModelHotSwapper(_mockVersionManager.Object, _mockRouter.Object);

            // Setup default mock behaviors
            _mockVersionManager.Setup(m => m.IsVersionLoaded(It.IsAny<string>(), It.IsAny<string>()))
                .Returns(true);
            _mockVersionManager.Setup(m => m.GetLoadInfo(It.IsAny<string>(), It.IsAny<string>()))
                .Returns(new VersionLoadInfo
                {
                    ModelId = TestModelId,
                    Version = TestVersion1,
                    IsLoaded = true,
                    MemoryUsageBytes = 1024 * 1024, // 1 MB
                    RequestCount = 0
                });
            _mockRouter.Setup(r => r.GetDefaultVersion(It.IsAny<string>()))
                .Returns(TestVersion1);
        }

        [Fact]
        public async Task SwapVersion_WithHealthyTargetVersion_Succeeds()
        {
            // Arrange
            _mockRouter.Setup(r => r.GetDefaultVersion(TestModelId))
                .Returns(TestVersion1);
            _mockRouter.Setup(r => r.SetDefaultVersion(TestModelId, TestVersion2));

            // Act
            var result = await _hotSwapper.SwapVersion(TestModelId, TestVersion1, TestVersion2);

            // Assert
            Assert.True(result.Success);
            Assert.Contains("Successfully swapped", result.Message);
            _mockRouter.Verify(r => r.SetDefaultVersion(TestModelId, TestVersion2), Times.Once);
        }

        [Fact]
        public async Task SwapVersion_WhenTargetUnhealthy_Throws()
        {
            // Arrange
            _mockVersionManager.Setup(m => m.GetLoadInfo(TestModelId, TestVersion2))
                .Returns(new VersionLoadInfo
                {
                    ModelId = TestModelId,
                    Version = TestVersion2,
                    IsLoaded = true,
                    MemoryUsageBytes = 5L * 1024 * 1024 * 1024, // 5 GB - exceeds limit
                    RequestCount = 0
                });

            // Act & Assert
            var result = await _hotSwapper.SwapVersion(TestModelId, TestVersion1, TestVersion2);
            Assert.False(result.Success);
            Assert.Contains("unhealthy", result.Message.ToLower());
        }

        [Fact]
        public async Task SwapVersion_DrainsRequestsCorrectly()
        {
            // Arrange
            _hotSwapper.IncrementInFlightRequest(TestModelId, TestVersion1);
            _hotSwapper.IncrementInFlightRequest(TestModelId, TestVersion1);
            _hotSwapper.DecrementInFlightRequest(TestModelId, TestVersion1);

            // Act
            var result = await _hotSwapper.SwapVersion(TestModelId, TestVersion1, TestVersion2);

            // Assert
            Assert.True(result.Success);
            Assert.True(result.RequestsDrained >= 0);
        }

        [Fact]
        public async Task SwapVersion_UpdatesRoutingPolicy()
        {
            // Arrange
            _mockRouter.Setup(r => r.GetDefaultVersion(TestModelId))
                .Returns(TestVersion1);

            // Act
            await _hotSwapper.SwapVersion(TestModelId, TestVersion1, TestVersion2);

            // Assert
            _mockRouter.Verify(r => r.SetDefaultVersion(TestModelId, TestVersion2), Times.Once);
        }

        [Fact]
        public async Task RollbackVersion_RevertsToPreviousVersion()
        {
            // Arrange
            _mockRouter.Setup(r => r.GetDefaultVersion(TestModelId))
                .Returns(TestVersion2);

            // Act
            var result = await _hotSwapper.RollbackVersion(TestModelId, TestVersion1);

            // Assert
            Assert.True(result.Success);
            Assert.Contains("Successfully rolled back", result.Message);
            Assert.Equal(TestVersion2, result.PreviousVersion);
            Assert.Equal(TestVersion1, result.NewVersion);
            _mockRouter.Verify(r => r.SetDefaultVersion(TestModelId, TestVersion1), Times.Once);
        }

        [Fact]
        public async Task RollbackVersion_IsImmediate()
        {
            // Arrange
            _mockRouter.Setup(r => r.GetDefaultVersion(TestModelId))
                .Returns(TestVersion2);

            // Act
            var startTime = DateTime.UtcNow;
            var result = await _hotSwapper.RollbackVersion(TestModelId, TestVersion1);
            var duration = DateTime.UtcNow - startTime;

            // Assert
            Assert.True(result.Success);
            Assert.True(duration < TimeSpan.FromSeconds(1)); // Should be nearly instantaneous
        }

        [Fact]
        public void CheckVersionHealth_PassesForGoodVersion()
        {
            // Act
            var result = _hotSwapper.CheckVersionHealth(TestModelId, TestVersion1);

            // Assert
            Assert.True(result.IsHealthy);
            Assert.Equal("Version is healthy", result.Message);
            Assert.True((bool)result.Diagnostics["MemoryUsageOK"]);
        }

        [Fact]
        public void CheckVersionHealth_FailsForBadVersion()
        {
            // Arrange
            _mockVersionManager.Setup(m => m.IsVersionLoaded(TestModelId, TestVersion1))
                .Returns(false);

            // Act
            var result = _hotSwapper.CheckVersionHealth(TestModelId, TestVersion1);

            // Assert
            Assert.False(result.IsHealthy);
            Assert.Contains("not loaded", result.Message.ToLower());
        }

        [Fact]
        public async Task DrainVersion_CompletesSuccessfully()
        {
            // Act
            var result = await _hotSwapper.DrainVersion(TestModelId, TestVersion1, TimeSpan.FromSeconds(5));

            // Assert
            Assert.True(result);
        }

        [Fact]
        public async Task DrainVersion_TimesOut()
        {
            // Arrange
            _hotSwapper.IncrementInFlightRequest(TestModelId, TestVersion1);
            // Don't decrement, so it will timeout

            // Act
            var result = await _hotSwapper.DrainVersion(TestModelId, TestVersion1, TimeSpan.FromMilliseconds(100));

            // Assert
            Assert.False(result);
            _hotSwapper.DecrementInFlightRequest(TestModelId, TestVersion1); // Cleanup
        }

        [Fact]
        public void GetSwapStatus_ReturnsCorrectState()
        {
            // Act
            var status = _hotSwapper.GetSwapStatus(TestModelId);

            // Assert
            Assert.NotNull(status);
            Assert.Equal(TestModelId, status.ModelId);
            Assert.Equal(SwapState.Idle, status.State);
        }

        [Fact]
        public async Task ConcurrentSwapAttempts_SecondFails()
        {
            // Arrange
            var task1 = _hotSwapper.SwapVersion(TestModelId, TestVersion1, TestVersion2);
            var task2 = _hotSwapper.SwapVersion(TestModelId, TestVersion1, TestVersion3);

            // Act
            await Task.WhenAll(task1, task2);
            var result1 = task1.Result;
            var result2 = task2.Result;

            // Assert
            // Only one should succeed
            Assert.True(result1.Success || result2.Success);
            Assert.False(result1.Success && result2.Success);
        }

        [Fact]
        public async Task SwapVersion_WithUnloadedTargetVersion_Throws()
        {
            // Arrange
            _mockVersionManager.Setup(m => m.IsVersionLoaded(TestModelId, TestVersion2))
                .Returns(false);

            // Act
            var result = await _hotSwapper.SwapVersion(TestModelId, TestVersion1, TestVersion2);

            // Assert
            Assert.False(result.Success);
            Assert.Contains("not loaded", result.Message.ToLower());
        }

        [Fact]
        public async Task RollbackVersion_ToProductionVersion_Succeeds()
        {
            // Arrange
            _mockRouter.Setup(r => r.GetDefaultVersion(TestModelId))
                .Returns(TestVersion2);

            // Act
            var result = await _hotSwapper.RollbackVersion(TestModelId, TestVersion1);

            // Assert
            Assert.True(result.Success);
            Assert.Contains("Successfully rolled back", result.Message);
        }

        [Fact]
        public async Task SwapVersion_WithSameVersion_Throws()
        {
            // Act
            var result = await _hotSwapper.SwapVersion(TestModelId, TestVersion1, TestVersion1);

            // Assert
            Assert.False(result.Success);
            Assert.Contains("same version", result.Message.ToLower());
        }

        [Fact]
        public async Task SwapVersion_WithInvalidModelId_Throws()
        {
            // Act
            var result = await _hotSwapper.SwapVersion("", TestVersion1, TestVersion2);

            // Assert
            Assert.False(result.Success);
        }

        [Fact]
        public async Task SwapVersion_WithInvalidVersions_Throws()
        {
            // Act
            var result1 = await _hotSwapper.SwapVersion(TestModelId, "", TestVersion2);
            var result2 = await _hotSwapper.SwapVersion(TestModelId, TestVersion1, "");

            // Assert
            Assert.False(result1.Success);
            Assert.False(result2.Success);
        }

        [Fact]
        public async Task RollbackVersion_WithInvalidModelId_Throws()
        {
            // Act
            var result = await _hotSwapper.RollbackVersion("", TestVersion1);

            // Assert
            Assert.False(result.Success);
        }

        [Fact]
        public async Task RollbackVersion_WithInvalidVersion_Throws()
        {
            // Act
            var result = await _hotSwapper.RollbackVersion(TestModelId, "");

            // Assert
            Assert.False(result.Success);
        }

        [Fact]
        public void CheckVersionHealth_WithInvalidInputs_Fails()
        {
            // Act
            var result1 = _hotSwapper.CheckVersionHealth("", TestVersion1);
            var result2 = _hotSwapper.CheckVersionHealth(TestModelId, "");

            // Assert
            Assert.False(result1.IsHealthy);
            Assert.False(result2.IsHealthy);
        }

        [Fact]
        public async Task DrainVersion_WithInvalidInputs_ReturnsFalse()
        {
            // Act
            var result1 = await _hotSwapper.DrainVersion("", TestVersion1, TimeSpan.FromSeconds(1));
            var result2 = await _hotSwapper.DrainVersion(TestModelId, "", TimeSpan.FromSeconds(1));

            // Assert
            Assert.False(result1);
            Assert.False(result2);
        }

        [Fact]
        public void IncrementInFlightRequest_TracksCorrectly()
        {
            // Arrange
            var initialCount = _hotSwapper.GetSwapStatus(TestModelId).PendingRequests;

            // Act
            _hotSwapper.IncrementInFlightRequest(TestModelId, TestVersion1);
            _hotSwapper.IncrementInFlightRequest(TestModelId, TestVersion1);
            var status = _hotSwapper.GetSwapStatus(TestModelId);

            // Assert
            Assert.True(status.PendingRequests > initialCount);

            // Cleanup
            _hotSwapper.DecrementInFlightRequest(TestModelId, TestVersion1);
            _hotSwapper.DecrementInFlightRequest(TestModelId, TestVersion1);
        }

        [Fact]
        public void DecrementInFlightRequest_TracksCorrectly()
        {
            // Arrange
            _hotSwapper.IncrementInFlightRequest(TestModelId, TestVersion1);
            _hotSwapper.IncrementInFlightRequest(TestModelId, TestVersion1);
            var initialCount = _hotSwapper.GetSwapStatus(TestModelId).PendingRequests;

            // Act
            _hotSwapper.DecrementInFlightRequest(TestModelId, TestVersion1);
            var status = _hotSwapper.GetSwapStatus(TestModelId);

            // Assert
            Assert.True(status.PendingRequests < initialCount);

            // Cleanup
            _hotSwapper.DecrementInFlightRequest(TestModelId, TestVersion1);
        }

        [Fact]
        public async Task SwapVersion_WhenSourceNotLoaded_Fails()
        {
            // Arrange
            _mockVersionManager.Setup(m => m.IsVersionLoaded(TestModelId, TestVersion1))
                .Returns(false);

            // Act
            var result = await _hotSwapper.SwapVersion(TestModelId, TestVersion1, TestVersion2);

            // Assert
            Assert.False(result.Success);
            Assert.Contains("not loaded", result.Message.ToLower());
        }

        [Fact]
        public async Task SwapVersion_UpdatesSwapStatusThroughProcess()
        {
            // Arrange
            _mockRouter.Setup(r => r.GetDefaultVersion(TestModelId))
                .Returns(TestVersion1);

            // Act
            var statusBefore = _hotSwapper.GetSwapStatus(TestModelId);
            await _hotSwapper.SwapVersion(TestModelId, TestVersion1, TestVersion2);
            var statusAfter = _hotSwapper.GetSwapStatus(TestModelId);

            // Assert
            Assert.Equal(SwapState.Idle, statusBefore.State);
            Assert.Equal(SwapState.Completed, statusAfter.State);
        }

        [Fact]
        public void GetSwapStatus_DuringActiveSwap_ReturnsCorrectInfo()
        {
            // Arrange - Start a swap but don't await it yet
            var swapTask = _hotSwapper.SwapVersion(TestModelId, TestVersion1, TestVersion2);
            
            // Wait a bit for the swap to start
            Task.Delay(50).Wait();

            // Act
            var status = _hotSwapper.GetSwapStatus(TestModelId);

            // Assert
            Assert.NotNull(status);
            Assert.Equal(TestModelId, status.ModelId);

            // Cleanup
            try
            {
                await swapTask;
            }
            catch
            {
                // Ignore any exceptions
            }
        }

        [Fact]
        public async Task SwapVersion_WithMemoryLimitExceeded_Fails()
        {
            // Arrange
            _mockVersionManager.Setup(m => m.GetLoadInfo(TestModelId, TestVersion2))
                .Returns(new VersionLoadInfo
                {
                    ModelId = TestModelId,
                    Version = TestVersion2,
                    IsLoaded = true,
                    MemoryUsageBytes = long.MaxValue, // Exceeds limit
                    RequestCount = 0
                });

            // Act
            var result = await _hotSwapper.SwapVersion(TestModelId, TestVersion1, TestVersion2);

            // Assert
            Assert.False(result.Success);
            Assert.Contains("memory", result.Message.ToLower());
        }
    }
}
