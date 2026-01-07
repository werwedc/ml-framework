using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.Pipeline;

namespace MLFramework.Tests.Pipeline
{
    public class GPipeSchedulerTests : IDisposable
    {
        private readonly IDevice _device;

        public GPipeSchedulerTests()
        {
            _device = DeviceManager.DefaultDevice;
        }

        public void Dispose()
        {
        }

        private GPipeScheduler CreateScheduler(int numStages)
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

            return new GPipeScheduler(stages, communicator, microBatchManager, config, checkpointManager);
        }

        [Fact]
        public void Constructor_WithValidInputs_CreatesScheduler()
        {
            // Arrange & Act
            var scheduler = CreateScheduler(2);

            // Assert
            Assert.NotNull(scheduler);
            Assert.Equal(2, scheduler.NumStages);
            Assert.NotNull(scheduler.Config);
        }

        [Fact]
        public async Task ForwardAsync_WithTwoStages_ProducesOutput()
        {
            // Arrange
            var scheduler = CreateScheduler(2);
            var input = Tensor.Zeros(new long[] { 8, 10 });

            // Act
            var output = await scheduler.ForwardAsync(input, microBatchIdx: 0);

            // Assert
            Assert.NotNull(output);
        }

        [Fact]
        public async Task BackwardAsync_WithTwoStages_ProducesGradients()
        {
            // Arrange
            var scheduler = CreateScheduler(2);
            var gradient = Tensor.Zeros(new long[] { 8, 10 });

            // Act
            var gradients = await scheduler.BackwardAsync(gradient, microBatchIdx: 0);

            // Assert
            Assert.NotNull(gradients);
            Assert.Equal(2, gradients.Count);
        }

        [Theory]
        [InlineData(2)]
        [InlineData(4)]
        [InlineData(8)]
        public async Task ForwardAsync_WithMultipleStages_WorksCorrectly(int numStages)
        {
            // Arrange
            var scheduler = CreateScheduler(numStages);
            var input = Tensor.Zeros(new long[] { 8, 10 });

            // Act & Assert - Should not throw
            await scheduler.ForwardAsync(input, microBatchIdx: 0);
        }

        [Fact]
        public async Task TrainIterationAsync_ExecutesSuccessfully()
        {
            // Arrange
            var scheduler = CreateScheduler(2);
            var input = Tensor.Zeros(new long[] { 32, 10 });
            var targets = Tensor.Zeros(new long[] { 32, 10 });

            Tensor lossFunction(Tensor output, Tensor target) => output;

            // Act & Assert - Should not throw
            await scheduler.TrainIterationAsync(input, targets, lossFunction);
        }

        [Fact]
        public void Reset_ClearsState()
        {
            // Arrange
            var scheduler = CreateScheduler(2);

            // Act & Assert - Should not throw
            scheduler.Reset();
        }

        [Fact]
        public async Task ForwardAsync_WithNullInput_ThrowsArgumentNullException()
        {
            // Arrange
            var scheduler = CreateScheduler(2);

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentNullException>(() =>
                scheduler.ForwardAsync(null!, microBatchIdx: 0));
        }

        [Fact]
        public async Task BackwardAsync_WithNullGradient_ThrowsArgumentNullException()
        {
            // Arrange
            var scheduler = CreateScheduler(2);

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentNullException>(() =>
                scheduler.BackwardAsync(null!, microBatchIdx: 0));
        }

        [Fact]
        public async Task TrainIterationAsync_WithNullInput_ThrowsArgumentNullException()
        {
            // Arrange
            var scheduler = CreateScheduler(2);
            var targets = Tensor.Zeros(new long[] { 32, 10 });
            Tensor lossFunction(Tensor output, Tensor target) => output;

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentNullException>(() =>
                scheduler.TrainIterationAsync(null!, targets, lossFunction));
        }

        [Fact]
        public void Dispose_CanBeCalledMultipleTimes()
        {
            // Arrange
            var scheduler = CreateScheduler(2);

            // Act - Should not throw
            scheduler.Dispose();
            scheduler.Dispose();

            // Assert - No exception
        }
    }
}
