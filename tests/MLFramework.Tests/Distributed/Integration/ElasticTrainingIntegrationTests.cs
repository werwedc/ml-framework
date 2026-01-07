using Microsoft.VisualStudio.TestTools.UnitTesting;
using MachineLearning.Distributed.Configuration;
using MachineLearning.Distributed.Coordinator;
using MachineLearning.Distributed.Data;
using MachineLearning.Distributed.Enums;
using MachineLearning.Distributed.Models;
using MachineLearning.Distributed.Scheduling;
using MachineLearning.Distributed.Worker;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace MLFramework.Tests.Distributed.Integration;

[TestClass]
public class ElasticTrainingIntegrationTests
{
    private ElasticTrainingConfig _config = null!;
    private ElasticCoordinator _coordinator = null!;
    private List<ElasticWorker> _workers = null!;
    private DataRedistributor _dataRedistributor = null!;
    private AdaptiveLearningRateScheduler _lrScheduler = null!;

    [TestInitialize]
    public void Setup()
    {
        _config = new ElasticTrainingConfig
        {
            MinWorkers = 1,
            MaxWorkers = 10,
            RescaleTimeoutMs = 1000,
            StabilityWindowMs = 500,
            WorkerHeartbeatTimeoutMs = 2000,
            FailureTolerancePercentage = 20,
            RedistributionType = RedistributionType.FullReshuffle
        };

        _coordinator = new ElasticCoordinator(_config);
        _workers = new List<ElasticWorker>();
        _dataRedistributor = new DataRedistributor(_config);
        _lrScheduler = new AdaptiveLearningRateScheduler(
            AdaptationStrategy.Linear,
            4, // initial worker count
            0.01f // initial learning rate
        );
    }

    [TestCleanup]
    public void TearDown()
    {
        foreach (var worker in _workers)
        {
            worker?.Dispose();
        }
        _coordinator?.Dispose();
    }

    [TestMethod]
    public async Task FullClusterLifecycle_WithMultipleWorkers_CompletesSuccessfully()
    {
        // Arrange
        var initialWorkerCount = 4;

        // Create and register workers
        for (int i = 0; i < initialWorkerCount; i++)
        {
            var workerId = new WorkerId($"worker{i}", "localhost", 8000 + i);
            var worker = new ElasticWorker(workerId, _config);
            _workers.Add(worker);

            var metadata = new WorkerMetadata
            {
                WorkerId = workerId,
                Status = WorkerStatus.Joining,
                JoinTime = DateTime.UtcNow,
                LastHeartbeat = DateTime.UtcNow,
                Rank = -1,
                LocalWorldSize = 1
            };

            _coordinator.RegisterWorker(metadata);
        }

        // Wait for stability
        await Task.Delay(_config.StabilityWindowMs + 100);

        // Act
        var clusterState = _coordinator.GetClusterState();

        // Assert
        Assert.AreEqual(initialWorkerCount, clusterState.WorldSize);
        Assert.AreEqual(initialWorkerCount, clusterState.Workers.Count);
    }

    [TestMethod]
    public async Task ScaleUpScenario_AddingWorkers_RescalesCorrectly()
    {
        // Arrange - Start with 2 workers
        for (int i = 0; i < 2; i++)
        {
            var workerId = new WorkerId($"worker{i}", "localhost", 8000 + i);
            var metadata = CreateWorkerMetadata(workerId);
            _coordinator.RegisterWorker(metadata);
        }

        await Task.Delay(_config.StabilityWindowMs + 100);
        var oldState = _coordinator.GetClusterState();

        // Act - Add 2 more workers
        for (int i = 2; i < 4; i++)
        {
            var workerId = new WorkerId($"worker{i}", "localhost", 8000 + i);
            var metadata = CreateWorkerMetadata(workerId);
            _coordinator.RegisterWorker(metadata);
        }

        await Task.Delay(_config.StabilityWindowMs + 100);
        var newState = _coordinator.GetClusterState();

        // Assert
        Assert.AreEqual(2, oldState.WorldSize);
        Assert.AreEqual(4, newState.WorldSize);
    }

    [TestMethod]
    public async Task ScaleDownScenario_RemovingWorkers_RescalesCorrectly()
    {
        // Arrange - Start with 4 workers
        for (int i = 0; i < 4; i++)
        {
            var workerId = new WorkerId($"worker{i}", "localhost", 8000 + i);
            var metadata = CreateWorkerMetadata(workerId);
            _coordinator.RegisterWorker(metadata);
        }

        await Task.Delay(_config.StabilityWindowMs + 100);
        var oldState = _coordinator.GetClusterState();

        // Act - Remove 2 workers
        var workerToRemove = new WorkerId("worker2", "localhost", 8002);
        _coordinator.UnregisterWorker(workerToRemove, graceful: true);

        workerToRemove = new WorkerId("worker3", "localhost", 8003);
        _coordinator.UnregisterWorker(workerToRemove, graceful: true);

        await Task.Delay(_config.StabilityWindowMs + 100);
        var newState = _coordinator.GetClusterState();

        // Assert
        Assert.AreEqual(4, oldState.WorldSize);
        Assert.AreEqual(2, newState.WorldSize);
    }

    [TestMethod]
    public void DataRedistribution_FullReshuffle_RedistributesAllShards()
    {
        // Arrange
        var oldTopology = new ClusterTopology
        {
            WorldSize = 2,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001)
            }
        };

        var newTopology = new ClusterTopology
        {
            WorldSize = 4,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001),
                new WorkerId("worker2", "localhost", 8002),
                new WorkerId("worker3", "localhost", 8003)
            }
        };

        // Act
        var plan = _dataRedistributor.CalculatePlan(oldTopology, newTopology, 100);

        // Assert
        Assert.IsTrue(plan.Transfers.Count > 0);
        Assert.AreEqual(100, plan.TotalShards);
        Assert.AreEqual(4, plan.WorkerAssignments.Count);
    }

    [TestMethod]
    public void DataRedistribution_Incremental_RestributesOnlyNecessaryShards()
    {
        // Arrange
        _config.RedistributionType = RedistributionType.Incremental;

        var oldTopology = new ClusterTopology
        {
            WorldSize = 2,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001)
            }
        };

        var newTopology = new ClusterTopology
        {
            WorldSize = 4,
            Workers = new List<WorkerId>
            {
                new WorkerId("worker0", "localhost", 8000),
                new WorkerId("worker1", "localhost", 8001),
                new WorkerId("worker2", "localhost", 8002),
                new WorkerId("worker3", "localhost", 8003)
            }
        };

        // Act
        var redistributor = new DataRedistributor(_config);
        var plan = redistributor.CalculatePlan(oldTopology, newTopology, 100);

        // Assert
        Assert.IsTrue(plan.Transfers.Count > 0);
        Assert.AreEqual(100, plan.TotalShards);
        // Incremental should have fewer transfers than full reshuffle
        Assert.IsTrue(plan.Transfers.Count < 100);
    }

    [TestMethod]
    public void LearningRateAdaptation_LinearScaling_ScalesProportionally()
    {
        // Arrange
        var initialLR = 0.01f;
        var initialWorkers = 4;
        var scheduler = new AdaptiveLearningRateScheduler(
            AdaptationStrategy.Linear,
            initialWorkers,
            initialLR
        );

        // Act - Scale up to 8 workers
        var newLR = scheduler.AdaptLearningRate(initialWorkers, 8, initialLR);

        // Assert
        Assert.AreEqual(0.02f, newLR, 0.0001f);

        // Act - Scale down to 2 workers
        newLR = scheduler.AdaptLearningRate(8, 2, newLR);

        // Assert
        Assert.AreEqual(0.005f, newLR, 0.0001f);
    }

    [TestMethod]
    public void LearningRateAdaptation_SquareRootScaling_ScalesWithSqrt()
    {
        // Arrange
        var initialLR = 0.01f;
        var initialWorkers = 4;
        var scheduler = new AdaptiveLearningRateScheduler(
            AdaptationStrategy.SquareRoot,
            initialWorkers,
            initialLR
        );

        // Act - Scale up to 16 workers (4x)
        var newLR = scheduler.AdaptLearningRate(initialWorkers, 16, initialLR);

        // Assert - Should be 2x (sqrt of 4x)
        Assert.AreEqual(0.02f, newLR, 0.0001f);
    }

    [TestMethod]
    public void LearningRateTransition_GeneratesCorrectInterpolatedValues()
    {
        // Arrange
        var oldLR = 0.01f;
        var newLR = 0.02f;
        var steps = 5;
        var scheduler = new AdaptiveLearningRateScheduler(
            AdaptationStrategy.Linear,
            4,
            0.01f
        );

        // Act
        var transition = scheduler.TransitionLearningRate(oldLR, newLR, steps).ToList();

        // Assert
        Assert.AreEqual(steps, transition.Count);
        Assert.AreEqual(oldLR, transition[0], 0.0001f);
        Assert.AreEqual(newLR, transition[transition.Count - 1], 0.0001f);
    }

    [TestMethod]
    public void ElasticDistributedSampler_DistributesDataCorrectly()
    {
        // Arrange
        var datasetSize = 100;
        var workerCount = 4;
        var samplers = new ElasticDistributedSampler[workerCount];

        for (int i = 0; i < workerCount; i++)
        {
            samplers[i] = new ElasticDistributedSampler(
                datasetSize,
                workerCount,
                i,
                allowDuplicate: false
            );
        }

        // Act
        var totalSamples = 0;
        var allSamples = new HashSet<int>();

        foreach (var sampler in samplers)
        {
            var samples = sampler.GetNumSamples();
            totalSamples += samples;

            var batch = sampler.GetNextBatch(samples).ToList();
            foreach (var sample in batch)
            {
                allSamples.Add(sample);
            }
        }

        // Assert
        Assert.AreEqual(datasetSize, totalSamples);
        Assert.AreEqual(datasetSize, allSamples.Count);
        // All samples should be unique
    }

    [TestMethod]
    public async Task EndToEndTraining_RescalingSimulation_MaintainsState()
    {
        // Arrange
        var workerCount = 4;
        for (int i = 0; i < workerCount; i++)
        {
            var workerId = new WorkerId($"worker{i}", "localhost", 8000 + i);
            var metadata = CreateWorkerMetadata(workerId);
            _coordinator.RegisterWorker(metadata);
        }

        await Task.Delay(_config.StabilityWindowMs + 100);

        // Simulate initial training state
        var initialState = _coordinator.GetGlobalState(10, 1000, 0.01f);

        // Act - Simulate scale up
        var newWorkerId = new WorkerId("worker4", "localhost", 8004);
        var newWorkerMetadata = CreateWorkerMetadata(newWorkerId);
        _coordinator.RegisterWorker(newWorkerMetadata);

        await Task.Delay(_config.StabilityWindowMs + 100);

        var newWorkerCount = _coordinator.GetClusterState().WorldSize;
        var newLR = _lrScheduler.AdaptLearningRate(workerCount, newWorkerCount, initialState.LearningRate);

        // Assert
        Assert.AreEqual(5, newWorkerCount);
        Assert.AreEqual(0.0125f, newLR, 0.0001f);
    }

    [TestMethod]
    public void FailureTolerance_ExceedingThreshold_ThrowsException()
    {
        // Arrange
        _config.MaxConsecutiveFailures = 2;
        _config.MinWorkers = 2;

        var coordinator = new ElasticCoordinator(_config);

        // Register 4 workers
        for (int i = 0; i < 4; i++)
        {
            var workerId = new WorkerId($"worker{i}", "localhost", 8000 + i);
            var metadata = CreateWorkerMetadata(workerId);
            coordinator.RegisterWorker(metadata);
        }

        // Act & Assert - Fail 2 workers (should be OK)
        coordinator.UnregisterWorker(new WorkerId("worker0", "localhost", 8000), graceful: false);
        coordinator.UnregisterWorker(new WorkerId("worker1", "localhost", 8001), graceful: false);

        // Act & Assert - Fail 3rd worker (should throw due to consecutive failures)
        Assert.ThrowsException<InvalidOperationException>(() =>
        {
            coordinator.UnregisterWorker(new WorkerId("worker2", "localhost", 8002), graceful: false);
        });

        coordinator.Dispose();
    }

    [TestMethod]
    public void StabilityWindow_RapidTopologyChanges_DoesNotTriggerRescaling()
    {
        // Arrange
        _config.StabilityWindowMs = 1000;
        var coordinator = new ElasticCoordinator(_config);

        var rescalingTriggered = false;
        coordinator.RescalingTriggered += _ => rescalingTriggered = true;

        // Act - Add and remove workers rapidly
        for (int i = 0; i < 3; i++)
        {
            var workerId = new WorkerId($"worker{i}", "localhost", 8000 + i);
            var metadata = CreateWorkerMetadata(workerId);
            coordinator.RegisterWorker(metadata);
        }

        // Wait less than stability window
        System.Threading.Thread.Sleep(500);

        // Remove a worker
        coordinator.UnregisterWorker(new WorkerId("worker0", "localhost", 8000), graceful: true);

        // Assert - Rescaling should not have been triggered yet
        Assert.IsFalse(rescalingTriggered);

        coordinator.Dispose();
    }

    #region Helper Methods

    private WorkerMetadata CreateWorkerMetadata(WorkerId workerId)
    {
        return new WorkerMetadata
        {
            WorkerId = workerId,
            Status = WorkerStatus.Joining,
            JoinTime = DateTime.UtcNow,
            LastHeartbeat = DateTime.UtcNow,
            Rank = -1,
            LocalWorldSize = 1
        };
    }

    #endregion
}
