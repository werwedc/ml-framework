# Spec: Elastic Training Integration Tests

## Overview
Implement integration tests that verify the end-to-end behavior of the elastic training system, testing the interaction between coordinator, workers, data redistribution, learning rate scheduling, and checkpointing.

## Deliverables

**File:** `tests/MachineLearning.Tests/Distributed/Integration/ElasticTrainingIntegrationTests.cs`
```csharp
namespace MachineLearning.Tests.Distributed.Integration;

using MachineLearning.Distributed.Configuration;
using MachineLearning.Distributed.Coordinator;
using MachineLearning.Distributed.Data;
using MachineLearning.Distributed.Models;
using MachineLearning.Distributed.Scheduling;
using MachineLearning.Distributed.Worker;

[TestFixture]
public class ElasticTrainingIntegrationTests
{
    private ElasticTrainingConfig _config = null!;
    private ElasticCoordinator _coordinator = null!;
    private List<ElasticWorker> _workers = null!;
    private DataRedistributor _dataRedistributor = null!;
    private AdaptiveLearningRateScheduler _lrScheduler = null!;

    [SetUp]
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

    [TearDown]
    public void TearDown()
    {
        foreach (var worker in _workers)
        {
            worker?.Dispose();
        }
        _coordinator?.Dispose();
    }

    [Test]
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
        Assert.That(clusterState.WorldSize, Is.EqualTo(initialWorkerCount));
        Assert.That(clusterState.Workers.Count, Is.EqualTo(initialWorkerCount));
    }

    [Test]
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
        Assert.That(oldState.WorldSize, Is.EqualTo(2));
        Assert.That(newState.WorldSize, Is.EqualTo(4));
    }

    [Test]
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
        Assert.That(oldState.WorldSize, Is.EqualTo(4));
        Assert.That(newState.WorldSize, Is.EqualTo(2));
    }

    [Test]
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
        Assert.That(plan.Transfers.Count, Is.GreaterThan(0));
        Assert.That(plan.TotalShards, Is.EqualTo(100));
        Assert.That(plan.WorkerAssignments.Count, Is.EqualTo(4));
    }

    [Test]
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
        var plan = _dataRedistributor.CalculatePlan(oldTopology, newTopology, 100);

        // Assert
        Assert.That(plan.Transfers.Count, Is.GreaterThan(0));
        Assert.That(plan.TotalShards, Is.EqualTo(100));
        // Incremental should have fewer transfers than full reshuffle
        Assert.That(plan.Transfers.Count, Is.LessThan(100));
    }

    [Test]
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
        Assert.That(newLR, Is.EqualTo(0.02f).Within(0.0001f));

        // Act - Scale down to 2 workers
        newLR = scheduler.AdaptLearningRate(8, 2, newLR);

        // Assert
        Assert.That(newLR, Is.EqualTo(0.005f).Within(0.0001f));
    }

    [Test]
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
        Assert.That(newLR, Is.EqualTo(0.02f).Within(0.0001f));
    }

    [Test]
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
        Assert.That(transition.Count, Is.EqualTo(steps));
        Assert.That(transition[0], Is.EqualTo(oldLR).Within(0.0001f));
        Assert.That(transition[^1], Is.EqualTo(newLR).Within(0.0001f));
    }

    [Test]
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
        Assert.That(totalSamples, Is.EqualTo(datasetSize));
        Assert.That(allSamples.Count, Is.EqualTo(datasetSize));
        // All samples should be unique
    }

    [Test]
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
        Assert.That(newWorkerCount, Is.EqualTo(5));
        Assert.That(newLR, Is.EqualTo(0.0125f).Within(0.0001f));
    }

    [Test]
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
        Assert.Throws<InvalidOperationException>(() =>
        {
            coordinator.UnregisterWorker(new WorkerId("worker2", "localhost", 8002), graceful: false);
        });

        coordinator.Dispose();
    }

    [Test]
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
        Thread.Sleep(500);

        // Remove a worker
        coordinator.UnregisterWorker(new WorkerId("worker0", "localhost", 8000), graceful: true);

        // Assert - Rescaling should not have been triggered yet
        Assert.That(rescalingTriggered, Is.False);

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
```

## Test Coverage

The integration tests should verify:

1. **Cluster Lifecycle**
   - Full lifecycle with multiple workers
   - Workers register and become active
   - Coordinator maintains correct cluster state

2. **Scaling Scenarios**
   - Scale up (adding workers)
   - Scale down (removing workers)
   - Rescaling triggers correctly

3. **Data Redistribution**
   - Full reshuffle redistributes all shards
   - Incremental redistribution minimizes transfers
   - Validation ensures correctness

4. **Learning Rate Adaptation**
   - Linear scaling works correctly
   - Square root scaling works correctly
   - Transitions generate correct interpolated values

5. **Data Sampling**
   - Distributed sampler divides data correctly
   - All samples are covered without duplication

6. **End-to-End Scenarios**
   - Training with rescaling maintains state
   - Workers coordinate properly during topology changes

7. **Failure Tolerance**
   - Tolerable failures don't abort training
   - Exceeding threshold throws exception

8. **Stability Mechanisms**
   - Stability window prevents rapid rescaling
   - Topology changes are aggregated properly

## Implementation Notes

1. Use NUnit testing framework
2. Tests are typically slower than unit tests due to timing
3. Use Task.Delay for stability window testing
4. Proper cleanup in TearDown
5. Helper methods reduce code duplication

## Dependencies
- NUnit testing framework
- All component implementations from previous specs

## Estimated Effort
~60 minutes

## Success Criteria
- All integration tests pass
- Tests verify realistic usage scenarios
- Components integrate correctly
- Performance characteristics are acceptable
- Edge cases are handled properly
