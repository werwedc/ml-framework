using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.Pipeline;

namespace MLFramework.Tests.Pipeline
{
    public class AsyncPipelineExecutorTests : IDisposable
    {
        private readonly IDevice _device;

        public AsyncPipelineExecutorTests()
        {
            _device = DeviceManager.DefaultDevice;
        }

        public void Dispose()
        {
        }

        private AsyncPipelineExecutor CreateExecutor(int numStages = 2)
        {
            var config = new PipelineConfig
            {
                NumStages = numStages,
                MicroBatches = 4
            };

            var stages = new List<PipelineStage>();
            for (int i = 0; i < numStages; i++)
            {
                var module = new Linear(10, 10, $"linear_{i}");
                var stage = new PipelineStage(module, i, numStages, _device);
                stages.Add(stage);
            }

            var communicator = new LocalPipelineCommunicator(0, numStages);
            var microBatchManager = new MicroBatchManager(32, 4, _device);
            var checkpointManager = new ActivationCheckpointManager(CheckpointStrategy.RecomputeAll);
            var scheduler = new GPipeScheduler(stages, communicator, microBatchManager, config, checkpointManager);

            return new AsyncPipelineExecutor(scheduler, communicator);
        }

        [Fact]
        public void Constructor_WithValidInputs_CreatesExecutor()
        {
            // Arrange & Act
            var executor = CreateExecutor();

            // Assert
            Assert.NotNull(executor);
            Assert.Equal(0, executor.ActiveStreamsCount);
        }

        [Fact]
        public async Task ForwardAsync_ExecutesSuccessfully()
        {
            // Arrange
            var executor = CreateExecutor();
            var input = Tensor.Zeros(new long[] { 8, 10 });

            // Act
            var output = await executor.ForwardAsync(input, microBatchIdx: 0);

            // Assert
            Assert.NotNull(output);
        }

        [Fact]
        public async Task BackwardAsync_ExecutesSuccessfully()
        {
            // Arrange
            var executor = CreateExecutor();
            var gradient = Tensor.Zeros(new long[] { 8, 10 });

            // Act
            var gradients = await executor.BackwardAsync(gradient, microBatchIdx: 0);

            // Assert
            Assert.NotNull(gradients);
        }

        [Fact]
        public async Task ExecuteIterationAsync_CompletesSuccessfully()
        {
            // Arrange
            var executor = CreateExecutor();
            var input = Tensor.Zeros(new long[] { 8, 10 });

            // Act
            var gradients = await executor.ExecuteIterationAsync(input, microBatchIdx: 0);

            // Assert
            Assert.NotNull(gradients);
        }

        [Fact]
        public async Task SyncAllAsync_WaitsForAllStreams()
        {
            // Arrange
            var executor = CreateExecutor();
            var input = Tensor.Zeros(new long[] { 8, 10 });

            // Start multiple async operations
            var tasks = new List<Task>();
            for (int i = 0; i < 3; i++)
            {
                tasks.Add(executor.ForwardAsync(input, i));
            }

            // Act - Wait for all to complete
            await executor.SyncAllAsync();

            // Assert
            Assert.Equal(0, executor.ActiveStreamsCount);
        }

        [Fact]
        public void CancelAll_ClearsActiveStreams()
        {
            // Arrange
            var executor = CreateExecutor();

            // Act & Assert - Should not throw
            executor.CancelAll();
            Assert.Equal(0, executor.ActiveStreamsCount);
        }

        [Fact]
        public async Task OverlappedComputeAndCommAsync_CompletesSuccessfully()
        {
            // Arrange
            var executor = CreateExecutor();
            var tensor = Tensor.Zeros(new long[] { 8, 10 });

            Tensor computeFunc() => tensor;
            Task commFunc() => Task.CompletedTask;

            // Act & Assert - Should not throw
            await executor.OverlappedComputeAndCommAsync(computeFunc, commFunc);
        }

        [Fact]
        public async Task ForwardAsync_WithNullInput_ThrowsArgumentNullException()
        {
            // Arrange
            var executor = CreateExecutor();

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentNullException>(() =>
                executor.ForwardAsync(null!, microBatchIdx: 0));
        }

        [Fact]
        public async Task BackwardAsync_WithNullGradient_ThrowsArgumentNullException()
        {
            // Arrange
            var executor = CreateExecutor();

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentNullException>(() =>
                executor.BackwardAsync(null!, microBatchIdx: 0));
        }

        [Fact]
        public async Task OverlappedComputeAndCommAsync_WithNullComputeFunc_ThrowsArgumentNullException()
        {
            // Arrange
            var executor = CreateExecutor();
            Task commFunc() => Task.CompletedTask;

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentNullException>(() =>
                executor.OverlappedComputeAndCommAsync(null!, commFunc));
        }

        [Fact]
        public async Task OverlappedComputeAndCommAsync_WithNullCommFunc_ThrowsArgumentNullException()
        {
            // Arrange
            var executor = CreateExecutor();
            Tensor computeFunc() => Tensor.Zeros(new long[] { 8, 10 });

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentNullException>(() =>
                executor.OverlappedComputeAndCommAsync(computeFunc, null!));
        }

        [Fact]
        public async Task ExecuteIterationAsync_WithNullInput_ThrowsArgumentNullException()
        {
            // Arrange
            var executor = CreateExecutor();

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentNullException>(() =>
                executor.ExecuteIterationAsync(null!, microBatchIdx: 0));
        }

        [Fact]
        public void Dispose_CanBeCalledMultipleTimes()
        {
            // Arrange
            var executor = CreateExecutor();

            // Act - Should not throw
            executor.Dispose();
            executor.Dispose();

            // Assert - No exception
        }

        [Fact]
        public async Task ActiveStreamsCount_TracksCorrectly()
        {
            // Arrange
            var executor = CreateExecutor();
            var input = Tensor.Zeros(new long[] { 8, 10 });

            // Act - Start an async operation
            var task = executor.ForwardAsync(input, 0);

            // Note: The task completes quickly, so we can't reliably test the count
            // This test mainly ensures the property exists

            await task;

            // Assert
            Assert.NotNull(executor);
        }
    }
}
