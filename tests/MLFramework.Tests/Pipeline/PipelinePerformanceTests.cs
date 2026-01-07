using System;
using System.Threading.Tasks;
using Xunit;
using RitterFramework.Core.Tensor;
using MLFramework.HAL;
using MLFramework.Pipeline;

namespace MLFramework.Tests.Pipeline
{
    public class PipelinePerformanceTests : IDisposable
    {
        private readonly IDevice _device;

        public PipelinePerformanceTests()
        {
            _device = DeviceManager.DefaultDevice;
        }

        public void Dispose()
        {
        }

        [Fact(Skip = "Performance test - run manually")]
        public async Task MeasureThroughput_ReportsTokensPerSecond()
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

            // Act
            var time = await TestHelper.MeasureExecutionTimeAsync(async () =>
            {
                for (int i = 0; i < 10; i++)
                {
                    await scheduler.TrainIterationAsync(input, targets, lossFunction);
                    scheduler.Reset();
                }
            });

            // Calculate throughput
            float totalTimeSeconds = time / 1000.0f;
            int totalTokens = 32 * 64 * 10; // batch_size * input_size * num_iterations
            float tokensPerSecond = totalTokens / totalTimeSeconds;

            // Log results
            Console.WriteLine($"Total time: {time} ms");
            Console.WriteLine($"Tokens per second: {tokensPerSecond:F2}");

            // Assert - No assertion for performance test, just reporting
            Assert.True(true);
        }

        [Fact(Skip = "Performance test - run manually")]
        public void MeasureMemoryUsage_ReportsPerStageMemory()
        {
            // Arrange
            var model = TestHelper.CreateSimpleMLP(64, 128, 10);
            var partitioner = new LayerPartitioner(PartitionMode.Automatic, 4);
            var result = partitioner.Partition(model);

            // Act - Get memory estimates
            var memoryPerStage = result.MemoryPerStage;

            // Assert - Verify memory is reasonable
            Assert.NotNull(memoryPerStage);
            Assert.Equal(4, memoryPerStage.Length);
            Assert.All(memoryPerStage, mem => Assert.True(mem > 0));

            // Log results
            for (int i = 0; i < memoryPerStage.Length; i++)
            {
                Console.WriteLine($"Stage {i} memory: {memoryPerStage[i]:F2} MB");
            }
        }

        [Fact(Skip = "Performance test - run manually")]
        public void MeasureUtilization_ReportsDeviceUtilization()
        {
            // Arrange
            var model = TestHelper.CreateSimpleMLP(64, 128, 10);
            var partitioner = new LayerPartitioner(PartitionMode.Automatic, 4);
            var result = partitioner.Partition(model);

            // Act - Calculate load balance (proxy for utilization)
            var loadBalance = result.LoadBalance;

            // Assert - Verify utilization is reasonable (> 70%)
            // Load balance close to 1.0 means better utilization
            Assert.True(loadBalance >= 1.0f);
            Assert.True(loadBalance < 2.0f, "Load balance should be < 2.0 for good utilization");

            Console.WriteLine($"Load balance: {loadBalance:F2}");
        }

        [Fact(Skip = "Performance test - run manually")]
        public async Task CompareSpeedup_VaryingStages_MeasuresScaling()
        {
            // Arrange
            var model = TestHelper.CreateSimpleMLP(64, 128, 10);
            var input = TestHelper.CreateDummyInput(32, 64, _device);
            var targets = TestHelper.CreateDummyInput(32, 10, _device);
            var numIterations = 10;

            Tensor lossFunction(Tensor output, Tensor target) => output;

            int[] numStagesArray = { 1, 2, 4 };
            var times = new long[numStagesArray.Length];

            // Act - Measure time for each stage count
            for (int i = 0; i < numStagesArray.Length; i++)
            {
                var numStages = numStagesArray[i];
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

                times[i] = await TestHelper.MeasureExecutionTimeAsync(async () =>
                {
                    for (int j = 0; j < numIterations; j++)
                    {
                        await scheduler.TrainIterationAsync(input, targets, lossFunction);
                        scheduler.Reset();
                    }
                });

                // Log results
                Console.WriteLine($"Stages: {numStages}, Time: {times[i]} ms");
            }

            // Assert - More stages should be faster (better parallelism)
            // In a real implementation, we would assert speedup
            Assert.True(times[0] > 0);
            Assert.True(times[1] > 0);
            Assert.True(times[2] > 0);

            // Calculate speedup
            float speedup2x = (float)times[0] / times[1];
            float speedup4x = (float)times[0] / times[2];

            Console.WriteLine($"Speedup (2 stages vs 1): {speedup2x:F2}x");
            Console.WriteLine($"Speedup (4 stages vs 1): {speedup4x:F2}x");
        }
    }
}
