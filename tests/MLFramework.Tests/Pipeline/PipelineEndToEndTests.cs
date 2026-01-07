using System;
using System.Threading.Tasks;
using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.Pipeline;

namespace MLFramework.Tests.Pipeline
{
    public class PipelineEndToEndTests : IDisposable
    {
        private readonly IDevice _device;

        public PipelineEndToEndTests()
        {
            _device = DeviceManager.DefaultDevice;
        }

        public void Dispose()
        {
        }

        [Fact]
        public async Task TrainSimpleMLP_WithPipeline_CompletesSuccessfully()
        {
            // Arrange
            var model = TestHelper.CreateSimpleMLP(64, 128, 10);
            var config = new PipelineConfig
            {
                NumStages = 2,
                MicroBatches = 4
            };

            var partitioner = new LayerPartitioner(PartitionMode.Uniform, 2);
            var partitionResult = partitioner.Partition(model);
            var communicator = new LocalPipelineCommunicator(0, 2);
            var microBatchManager = new MicroBatchManager(32, 4, _device);
            var checkpointManager = new ActivationCheckpointManager(CheckpointStrategy.StoreAll);

            var scheduler = new GPipeScheduler(
                partitionResult.Stages,
                communicator,
                microBatchManager,
                config,
                checkpointManager);

            var input = TestHelper.CreateDummyInput(32, 64, _device);
            var targets = TestHelper.CreateDummyInput(32, 10, _device);

            Tensor lossFunction(Tensor output, Tensor target) => output;

            // Act & Assert - Should complete without throwing
            for (int i = 0; i < 3; i++)
            {
                await scheduler.TrainIterationAsync(input, targets, lossFunction);
                scheduler.Reset();
            }
        }

        [Theory]
        [InlineData(2)]
        [InlineData(4)]
        [InlineData(8)]
        public async Task TrainModel_WithDifferentNumStages_WorksCorrectly(int numStages)
        {
            // Arrange
            var model = TestHelper.CreateSimpleMLP(64, 128, 10);
            var config = new PipelineConfig
            {
                NumStages = numStages,
                MicroBatches = 4
            };

            var partitioner = new LayerPartitioner(PartitionMode.Uniform, numStages);
            var partitionResult = partitioner.Partition(model);
            var communicator = new LocalPipelineCommunicator(0, numStages);
            var microBatchManager = new MicroBatchManager(32, 4, _device);
            var checkpointManager = new ActivationCheckpointManager(CheckpointStrategy.StoreAll);

            var scheduler = new GPipeScheduler(
                partitionResult.Stages,
                communicator,
                microBatchManager,
                config,
                checkpointManager);

            var input = TestHelper.CreateDummyInput(32, 64, _device);
            var targets = TestHelper.CreateDummyInput(32, 10, _device);

            Tensor lossFunction(Tensor output, Tensor target) => output;

            // Act & Assert - Should complete without throwing
            await scheduler.TrainIterationAsync(input, targets, lossFunction);
        }

        [Theory]
        [InlineData(2)]
        [InlineData(4)]
        [InlineData(8)]
        public async Task TrainModel_WithDifferentMicroBatches_WorksCorrectly(int microBatches)
        {
            // Arrange
            var model = TestHelper.CreateSimpleMLP(64, 128, 10);
            var config = new PipelineConfig
            {
                NumStages = 2,
                MicroBatches = microBatches
            };

            var partitioner = new LayerPartitioner(PartitionMode.Uniform, 2);
            var partitionResult = partitioner.Partition(model);
            var communicator = new LocalPipelineCommunicator(0, 2);
            var microBatchManager = new MicroBatchManager(32, microBatches, _device);
            var checkpointManager = new ActivationCheckpointManager(CheckpointStrategy.StoreAll);

            var scheduler = new GPipeScheduler(
                partitionResult.Stages,
                communicator,
                microBatchManager,
                config,
                checkpointManager);

            var input = TestHelper.CreateDummyInput(32, 64, _device);
            var targets = TestHelper.CreateDummyInput(32, 10, _device);

            Tensor lossFunction(Tensor output, Tensor target) => output;

            // Act & Assert - Should complete without throwing
            await scheduler.TrainIterationAsync(input, targets, lossFunction);
        }
    }
}
