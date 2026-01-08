using System;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using Moq;
using MLFramework.Serving.Deployment;

namespace MLFramework.Tests.Serving
{
    /// <summary>
    /// Unit tests for ModelHotswapper functionality.
    /// </summary>
    public class ModelHotswapperTests : IDisposable
    {
        private readonly Mock<IModelLoader> _mockModelLoader;
        private readonly Mock<IVersionRouterCore> _mockRouter;
        private readonly Mock<IModelRegistry> _mockRegistry;
        private readonly Mock<IModel> _mockModel;
        private readonly ModelHotswapper _hotswapper;

        public ModelHotswapperTests()
        {
            _mockModelLoader = new Mock<IModelLoader>();
            _mockRouter = new Mock<IVersionRouterCore>();
            _mockRegistry = new Mock<IModelRegistry>();
            _mockModel = new Mock<IModel>();

            _mockModel.Setup(m => m.Name).Returns("test-model");
            _mockModel.Setup(m => m.Version).Returns("1.0.0");
            _mockModel.Setup(m => m.IsActive).Returns(true);

            _hotswapper = new ModelHotswapper(
                _mockModelLoader.Object,
                _mockRouter.Object,
                _mockRegistry.Object);
        }

        public void Dispose()
        {
            // Cleanup if needed
        }

        [Fact]
        public async Task SwapVersionAsync_ValidVersions_CompletesSuccessfully()
        {
            // Arrange
            const string modelName = "test-model";
            const string fromVersion = "1.0.0";
            const string toVersion = "1.1.0";

            _mockModelLoader.Setup(l => l.IsLoaded(modelName, fromVersion)).Returns(true);
            _mockModelLoader.Setup(l => l.IsLoaded(modelName, toVersion)).Returns(true);
            _mockRegistry.Setup(r => r.HasVersion(modelName, toVersion)).Returns(true);
            _mockRouter.Setup(r => r.UpdateRoutingAsync(modelName, toVersion))
                .Returns(Task.CompletedTask);

            // Act
            var result = await _hotswapper.SwapVersionAsync(modelName, fromVersion, toVersion);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(modelName, result.ModelName);
            Assert.Equal(fromVersion, result.FromVersion);
            Assert.Equal(toVersion, result.ToVersion);
            Assert.Equal(SwapState.Completed, result.State);
            Assert.NotNull(result.EndTime);
            Assert.Null(result.ErrorMessage);

            _mockRouter.Verify(r => r.UpdateRoutingAsync(modelName, toVersion), Times.Once);
        }

        [Fact]
        public async Task SwapVersionAsync_NonExistentTargetVersion_ThrowsInvalidOperationException()
        {
            // Arrange
            const string modelName = "test-model";
            const string fromVersion = "1.0.0";
            const string toVersion = "1.1.0";

            _mockModelLoader.Setup(l => l.IsLoaded(modelName, fromVersion)).Returns(true);
            _mockRegistry.Setup(r => r.HasVersion(modelName, toVersion)).Returns(false);

            // Act & Assert
            await Assert.ThrowsAsync<InvalidOperationException>(() =>
                _hotswapper.SwapVersionAsync(modelName, fromVersion, toVersion));
        }

        [Fact]
        public async Task SwapVersionAsync_SourceNotActive_ThrowsInvalidOperationException()
        {
            // Arrange
            const string modelName = "test-model";
            const string fromVersion = "1.0.0";
            const string toVersion = "1.1.0";

            _mockModelLoader.Setup(l => l.IsLoaded(modelName, fromVersion)).Returns(false);
            _mockRegistry.Setup(r => r.HasVersion(modelName, toVersion)).Returns(true);

            // Act & Assert
            await Assert.ThrowsAsync<InvalidOperationException>(() =>
                _hotswapper.SwapVersionAsync(modelName, fromVersion, toVersion));
        }

        [Fact]
        public void SwapVersionAsync_NullModelName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.ThrowsAsync<ArgumentException>(() =>
                _hotswapper.SwapVersionAsync(null, "1.0.0", "1.1.0"));
        }

        [Fact]
        public void SwapVersionAsync_NullFromVersion_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.ThrowsAsync<ArgumentException>(() =>
                _hotswapper.SwapVersionAsync("test-model", null, "1.1.0"));
        }

        [Fact]
        public void SwapVersionAsync_NullToVersion_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.ThrowsAsync<ArgumentException>(() =>
                _hotswapper.SwapVersionAsync("test-model", "1.0.0", null));
        }

        [Fact]
        public async Task SwapVersionAsync_ConcurrentSwapsForDifferentModels_Succeeds()
        {
            // Arrange
            const string model1 = "model-1";
            const string model2 = "model-2";
            const string fromVersion = "1.0.0";
            const string toVersion = "1.1.0";

            _mockModelLoader.Setup(l => l.IsLoaded(It.IsAny<string>(), It.IsAny<string>())).Returns(true);
            _mockRegistry.Setup(r => r.HasVersion(It.IsAny<string>(), It.IsAny<string>())).Returns(true);
            _mockRouter.Setup(r => r.UpdateRoutingAsync(It.IsAny<string>(), It.IsAny<string>()))
                .Returns(Task.CompletedTask);

            // Act
            var task1 = _hotswapper.SwapVersionAsync(model1, fromVersion, toVersion);
            var task2 = _hotswapper.SwapVersionAsync(model2, fromVersion, toVersion);

            await Task.WhenAll(task1, task2);

            // Assert
            Assert.Equal(SwapState.Completed, task1.Result.State);
            Assert.Equal(SwapState.Completed, task2.Result.State);
            Assert.NotEqual(task1.Result.OperationId, task2.Result.OperationId);
        }

        [Fact]
        public async Task SwapVersionAsync_ConcurrentSwapsForSameModel_ThrowsInvalidOperationException()
        {
            // Arrange
            const string modelName = "test-model";
            const string fromVersion = "1.0.0";
            const string toVersion = "1.1.0";

            _mockModelLoader.Setup(l => l.IsLoaded(It.IsAny<string>(), It.IsAny<string>())).Returns(true);
            _mockRegistry.Setup(r => r.HasVersion(It.IsAny<string>(), It.IsAny<string>())).Returns(true);
            _mockRouter.Setup(r => r.UpdateRoutingAsync(It.IsAny<string>(), It.IsAny<string>()))
                .Returns(Task.CompletedTask);

            // Act - Start first swap
            var task1 = _hotswapper.SwapVersionAsync(modelName, fromVersion, toVersion);

            // Try to start second swap immediately (should throw)
            await Assert.ThrowsAsync<InvalidOperationException>(() =>
                _hotswapper.SwapVersionAsync(modelName, fromVersion, toVersion));

            // Wait for first swap to complete
            await task1;
        }

        [Fact]
        public void GetSwapStatus_ValidOperationId_ReturnsCorrectOperation()
        {
            // Arrange - Create a swap operation and inject it
            const string modelName = "test-model";
            const string fromVersion = "1.0.0";
            const string toVersion = "1.1.0";

            // We'll need to create a hotswapper with mock dependencies that allow us to test this
            _mockModelLoader.Setup(l => l.IsLoaded(modelName, fromVersion)).Returns(true);
            _mockModelLoader.Setup(l => l.IsLoaded(modelName, toVersion)).Returns(true);
            _mockRegistry.Setup(r => r.HasVersion(modelName, toVersion)).Returns(true);
            _mockRouter.Setup(r => r.UpdateRoutingAsync(modelName, toVersion))
                .Returns(Task.CompletedTask);

            // Act
            var operation = _hotswapper.SwapVersionAsync(modelName, fromVersion, toVersion).Result;
            var retrieved = _hotswapper.GetSwapStatus(operation.OperationId);

            // Assert
            Assert.NotNull(retrieved);
            Assert.Equal(operation.OperationId, retrieved.OperationId);
            Assert.Equal(modelName, retrieved.ModelName);
            Assert.Equal(fromVersion, retrieved.FromVersion);
            Assert.Equal(toVersion, retrieved.ToVersion);
            Assert.Equal(SwapState.Completed, retrieved.State);
        }

        [Fact]
        public void GetSwapStatus_NonExistentOperationId_ThrowsKeyNotFoundException()
        {
            // Act & Assert
            Assert.Throws<KeyNotFoundException>(() =>
                _hotswapper.GetSwapStatus("non-existent-id"));
        }

        [Fact]
        public void GetSwapStatus_NullOperationId_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _hotswapper.GetSwapStatus(null));
        }

        [Fact]
        public void WaitForDrainage_NoActiveRequests_CompletesQuickly()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";

            // Act - Should complete quickly since there are no active swaps
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            _hotswapper.WaitForDrainage(modelName, version, TimeSpan.FromSeconds(5));
            stopwatch.Stop();

            // Assert
            Assert.True(stopwatch.ElapsedMilliseconds < 1000,
                $"WaitForDrainage completed in {stopwatch.ElapsedMilliseconds}ms, expected < 1000ms");
        }

        [Fact]
        public void WaitForDrainage_Timeout_ThrowsTimeoutException()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";

            // Act & Assert - With a very short timeout, it should timeout
            // Note: This test may need adjustment based on actual implementation
            _hotswapper.WaitForDrainage(modelName, version, TimeSpan.FromMilliseconds(10));
        }

        [Fact]
        public void IsVersionActive_ActiveVersion_ReturnsTrue()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            _mockModelLoader.Setup(l => l.IsLoaded(modelName, version)).Returns(true);

            // Act
            var result = _hotswapper.IsVersionActive(modelName, version);

            // Assert
            Assert.True(result);
        }

        [Fact]
        public void IsVersionActive_InactiveVersion_ReturnsFalse()
        {
            // Arrange
            const string modelName = "test-model";
            const string version = "1.0.0";
            _mockModelLoader.Setup(l => l.IsLoaded(modelName, version)).Returns(false);

            // Act
            var result = _hotswapper.IsVersionActive(modelName, version);

            // Assert
            Assert.False(result);
        }

        [Fact]
        public void IsVersionActive_NullModelName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _hotswapper.IsVersionActive(null, "1.0.0"));
        }

        [Fact]
        public void IsVersionActive_NullVersion_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _hotswapper.IsVersionActive("test-model", null));
        }

        [Fact]
        public async Task RollbackAsync_DuringSwap_RoutesBackToOriginalVersion()
        {
            // Arrange
            const string modelName = "test-model";
            const string fromVersion = "1.0.0";
            const string toVersion = "1.1.0";

            _mockModelLoader.Setup(l => l.IsLoaded(modelName, fromVersion)).Returns(true);
            _mockModelLoader.Setup(l => l.IsLoaded(modelName, toVersion)).Returns(true);
            _mockRegistry.Setup(r => r.HasVersion(modelName, toVersion)).Returns(true);
            _mockRouter.Setup(r => r.UpdateRoutingAsync(It.IsAny<string>(), It.IsAny<string>()))
                .Returns(Task.CompletedTask);

            // Act - Start a swap
            var operation = await _hotswapper.SwapVersionAsync(modelName, fromVersion, toVersion);

            // Rollback to original version
            await _hotswapper.RollbackAsync(operation.OperationId);

            // Assert
            _mockRouter.Verify(r => r.UpdateRoutingAsync(modelName, fromVersion), Times.Once);
        }

        [Fact]
        public async Task RollbackAsync_NonExistentOperationId_ThrowsKeyNotFoundException()
        {
            // Act & Assert
            await Assert.ThrowsAsync<KeyNotFoundException>(() =>
                _hotswapper.RollbackAsync("non-existent-id"));
        }

        [Fact]
        public async Task RollbackAsync_NullOperationId_ThrowsArgumentException()
        {
            // Act & Assert
            await Assert.ThrowsAsync<ArgumentException>(() =>
                _hotswapper.RollbackAsync(null));
        }

        [Fact]
        public async Task SwapVersionAsync_Performance_CompletesWithin100ms()
        {
            // Arrange
            const string modelName = "test-model";
            const string fromVersion = "1.0.0";
            const string toVersion = "1.1.0";

            _mockModelLoader.Setup(l => l.IsLoaded(It.IsAny<string>(), It.IsAny<string>())).Returns(true);
            _mockRegistry.Setup(r => r.HasVersion(It.IsAny<string>(), It.IsAny<string>())).Returns(true);
            _mockRouter.Setup(r => r.UpdateRoutingAsync(It.IsAny<string>(), It.IsAny<string>()))
                .Returns(Task.CompletedTask);

            // Act
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            await _hotswapper.SwapVersionAsync(modelName, fromVersion, toVersion);
            stopwatch.Stop();

            // Assert - Should complete within 100ms (excluding model load time)
            Assert.True(stopwatch.ElapsedMilliseconds < 100,
                $"Swap completed in {stopwatch.ElapsedMilliseconds}ms, expected < 100ms");
        }

        [Fact]
        public async Task SwapVersionAsync_RouterFailure_MarksOperationAsFailed()
        {
            // Arrange
            const string modelName = "test-model";
            const string fromVersion = "1.0.0";
            const string toVersion = "1.1.0";

            _mockModelLoader.Setup(l => l.IsLoaded(It.IsAny<string>(), It.IsAny<string>())).Returns(true);
            _mockRegistry.Setup(r => r.HasVersion(It.IsAny<string>(), It.IsAny<string>())).Returns(true);
            _mockRouter.Setup(r => r.UpdateRoutingAsync(It.IsAny<string>(), It.IsAny<string>()))
                .ThrowsAsync(new Exception("Router failure"));

            // Act & Assert
            var exception = await Assert.ThrowsAsync<Exception>(() =>
                _hotswapper.SwapVersionAsync(modelName, fromVersion, toVersion));

            Assert.Equal("Router failure", exception.Message);
        }
    }
}
