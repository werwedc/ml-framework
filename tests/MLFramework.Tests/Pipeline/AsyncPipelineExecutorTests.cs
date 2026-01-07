using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.HAL.CUDA;
using MLFramework.Pipeline;

namespace MLFramework.Tests.Pipeline
{
    public class AsyncPipelineExecutorTests : IDisposable
    {
        private readonly CudaDevice _device;

        public AsyncPipelineExecutorTests()
        {
            // Use a CUDA device if available
            _device = new CudaDevice(0);
        }

        public void Dispose()
        {
            _device.Dispose();
        }

        private AsyncPipelineExecutor CreateExecutor(int numStages = 2, int numStreams = 2)
        {
            var module = new TestHelper.Linear(10, 10, $"linear_stage_0");
            var stage = new PipelineStage(module, 0, numStages, _device);
            var communicator = new LocalPipelineCommunicator(0, numStages);

            return new AsyncPipelineExecutor(stage, communicator, numStreams);
        }

        [Fact]
        public void Constructor_WithValidInputs_CreatesExecutor()
        {
            // Arrange & Act
            var executor = CreateExecutor();

            // Assert
            Assert.NotNull(executor);
            Assert.Equal(2, executor.NumComputeStreams);
            Assert.Equal(2, executor.NumCommStreams);
            Assert.Equal(0, executor.ActiveOperationsCount);
        }

        [Fact]
        public void Constructor_WithCustomStreamCount_CreatesExecutor()
        {
            // Arrange & Act
            var executor = CreateExecutor(numStreams: 4);

            // Assert
            Assert.NotNull(executor);
            Assert.Equal(4, executor.NumComputeStreams);
            Assert.Equal(4, executor.NumCommStreams);
        }

        [Fact]
        public void Constructor_WithNullStage_ThrowsArgumentNullException()
        {
            // Arrange
            var communicator = new LocalPipelineCommunicator(0, 2);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new AsyncPipelineExecutor(null!, communicator));
        }

        [Fact]
        public void Constructor_WithNullCommunicator_ThrowsArgumentNullException()
        {
            // Arrange
            var module = new Linear(10, 10);
            var stage = new PipelineStage(module, 0, 2, _device);

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new AsyncPipelineExecutor(stage, null!));
        }

        [Fact]
        public void Constructor_WithInvalidStreamCount_ThrowsArgumentException()
        {
            // Arrange
            var module = new Linear(10, 10);
            var stage = new PipelineStage(module, 0, 2, _device);
            var communicator = new LocalPipelineCommunicator(0, 2);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new AsyncPipelineExecutor(stage, communicator, 0));

            Assert.Throws<ArgumentException>(() =>
                new AsyncPipelineExecutor(stage, communicator, -1));
        }

        [Fact]
        public async Task ForwardAsync_ExecutesSuccessfully()
        {
            // Arrange
            var executor = CreateExecutor();
            var input = Tensor.Zeros(new long[] { 8, 10 });

            // Act
            var output = await executor.ForwardAsync(input, streamIndex: 0);

            // Assert
            Assert.NotNull(output);
            Assert.Equal(input.Shape, output.Shape);
        }

        [Fact]
        public async Task ForwardAsync_WithNullInput_ThrowsArgumentNullException()
        {
            // Arrange
            var executor = CreateExecutor();

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentNullException>(() =>
                executor.ForwardAsync(null!, streamIndex: 0));
        }

        [Fact]
        public async Task BackwardAsync_ExecutesSuccessfully()
        {
            // Arrange
            var executor = CreateExecutor();
            var gradient = Tensor.Zeros(new long[] { 8, 10 });

            // Act
            var result = await executor.BackwardAsync(gradient, streamIndex: 0);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(gradient.Shape, result.Shape);
        }

        [Fact]
        public async Task BackwardAsync_WithNullGradient_ThrowsArgumentNullException()
        {
            // Arrange
            var executor = CreateExecutor();

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentNullException>(() =>
                executor.BackwardAsync(null!, streamIndex: 0));
        }

        [Fact]
        public async Task SendAsync_ExecutesSuccessfully()
        {
            // Arrange
            var executor = CreateExecutor();
            var tensor = Tensor.Zeros(new long[] { 8, 10 });

            // Act & Assert - Should not throw
            var result = await executor.SendAsync(tensor, destinationRank: 1, streamIndex: 0);
            Assert.NotNull(result);
        }

        [Fact]
        public async Task SendAsync_WithNullTensor_ThrowsArgumentNullException()
        {
            // Arrange
            var executor = CreateExecutor();

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentNullException>(() =>
                executor.SendAsync(null!, destinationRank: 1, streamIndex: 0));
        }

        [Fact]
        public async Task ReceiveAsync_ExecutesSuccessfully()
        {
            // Arrange
            var executor = CreateExecutor();

            // Act & Assert - Should not throw (may timeout depending on communicator implementation)
            try
            {
                var result = await executor.ReceiveAsync(sourceRank: 1, streamIndex: 0);
                Assert.NotNull(result);
            }
            catch (TimeoutException)
            {
                // Expected if no peer is sending
            }
        }

        [Fact]
        public async Task SyncComputeAsync_CompletesSuccessfully()
        {
            // Arrange
            var executor = CreateExecutor();
            var input = Tensor.Zeros(new long[] { 8, 10 });

            // Act - Start a forward pass
            var task = executor.ForwardAsync(input, streamIndex: 0);

            // Wait for compute streams to sync
            await executor.SyncComputeAsync();

            await task;

            // Assert - No exception thrown
        }

        [Fact]
        public async Task SyncCommAsync_CompletesSuccessfully()
        {
            // Arrange
            var executor = CreateExecutor();

            // Act & Assert - Should not throw
            await executor.SyncCommAsync();
        }

        [Fact]
        public async Task SyncAllAsync_WaitsForAllOperations()
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
            Assert.Equal(0, executor.ActiveOperationsCount);
        }

        [Fact]
        public void GetComputeStream_ReturnsValidStream()
        {
            // Arrange
            var executor = CreateExecutor();

            // Act
            var stream = executor.GetComputeStream(0);

            // Assert
            Assert.NotNull(stream);
        }

        [Fact]
        public void GetCommStream_ReturnsValidStream()
        {
            // Arrange
            var executor = CreateExecutor();

            // Act
            var stream = executor.GetCommStream(0);

            // Assert
            Assert.NotNull(stream);
        }

        [Fact]
        public void GetComputeStream_RoundRobinAssignment()
        {
            // Arrange
            var executor = CreateExecutor(numStreams: 2);

            // Act
            var stream1 = executor.GetComputeStream(0);
            var stream2 = executor.GetComputeStream(1);
            var stream3 = executor.GetComputeStream(2); // Should wrap around to stream 0

            // Assert
            Assert.NotNull(stream1);
            Assert.NotNull(stream2);
            Assert.NotNull(stream3);
            Assert.Equal(stream1, stream3); // Round-robin wraps
        }

        [Fact]
        public void GetCommStream_RoundRobinAssignment()
        {
            // Arrange
            var executor = CreateExecutor(numStreams: 2);

            // Act
            var stream1 = executor.GetCommStream(0);
            var stream2 = executor.GetCommStream(1);
            var stream3 = executor.GetCommStream(2); // Should wrap around to stream 0

            // Assert
            Assert.NotNull(stream1);
            Assert.NotNull(stream2);
            Assert.NotNull(stream3);
            Assert.Equal(stream1, stream3); // Round-robin wraps
        }

        [Fact]
        public void GetComputeStreamByIndex_WithValidIndex_ReturnsStream()
        {
            // Arrange
            var executor = CreateExecutor(numStreams: 2);

            // Act
            var stream = executor.GetComputeStreamByIndex(1);

            // Assert
            Assert.NotNull(stream);
        }

        [Fact]
        public void GetComputeStreamByIndex_WithInvalidIndex_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var executor = CreateExecutor(numStreams: 2);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                executor.GetComputeStreamByIndex(2));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                executor.GetComputeStreamByIndex(-1));
        }

        [Fact]
        public void GetCommStreamByIndex_WithValidIndex_ReturnsStream()
        {
            // Arrange
            var executor = CreateExecutor(numStreams: 2);

            // Act
            var stream = executor.GetCommStreamByIndex(1);

            // Assert
            Assert.NotNull(stream);
        }

        [Fact]
        public void GetCommStreamByIndex_WithInvalidIndex_ThrowsArgumentOutOfRangeException()
        {
            // Arrange
            var executor = CreateExecutor(numStreams: 2);

            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                executor.GetCommStreamByIndex(2));

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                executor.GetCommStreamByIndex(-1));
        }

        [Fact]
        public async Task ForwardAsync_WithDifferentStreams_ExecutesSuccessfully()
        {
            // Arrange
            var executor = CreateExecutor(numStreams: 4);
            var input = Tensor.Zeros(new long[] { 8, 10 });

            // Act
            var tasks = new List<Task<Tensor>>();
            for (int i = 0; i < 4; i++)
            {
                tasks.Add(executor.ForwardAsync(input, streamIndex: i));
            }

            await Task.WhenAll(tasks);

            // Assert
            Assert.All(tasks, t => Assert.NotNull(t.Result));
        }

        [Fact]
        public void Dispose_AfterUse_DoesNotThrow()
        {
            // Arrange
            var executor = CreateExecutor();

            // Act & Assert - Should not throw
            executor.Dispose();
            executor.Dispose(); // Can be called multiple times
        }

        [Fact]
        public async Task AfterDispose_ThrowsObjectDisposedException()
        {
            // Arrange
            var executor = CreateExecutor();
            var input = Tensor.Zeros(new long[] { 8, 10 });
            executor.Dispose();

            // Act & Assert
            await Assert.ThrowsAsync<ObjectDisposedException>(() =>
                executor.ForwardAsync(input, streamIndex: 0));

            Assert.Throws<ObjectDisposedException>(() =>
                executor.GetComputeStream(0));

            await Assert.ThrowsAsync<ObjectDisposedException>(() =>
                executor.SyncAllAsync());
        }
    }
}
