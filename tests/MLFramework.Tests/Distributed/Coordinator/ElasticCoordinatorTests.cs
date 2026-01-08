using MachineLearning.Distributed.Coordinator;
using MachineLearning.Distributed.Configuration;
using MachineLearning.Distributed.Enums;
using MachineLearning.Distributed.Models;

namespace MLFramework.Tests.Distributed.Coordinator;

/// <summary>
/// Unit tests for ElasticCoordinator
/// </summary>
public class ElasticCoordinatorTests : IDisposable
{
    private ElasticCoordinator? _coordinator;
    private ElasticTrainingConfig? _config;

    [SetUp]
    public void Setup()
    {
        _config = new ElasticTrainingConfig
        {
            MinWorkers = 1,
            MaxWorkers = 8,
            StabilityWindowMs = 500,
            WorkerHeartbeatTimeoutMs = 1000,
            MaxConsecutiveFailures = 3,
            FailureTolerancePercentage = 20
        };

        _coordinator = new ElasticCoordinator(_config);
    }

    [TearDown]
    public void TearDown()
    {
        _coordinator?.Dispose();
    }

    public void Dispose()
    {
        _coordinator?.Dispose();
    }

    private WorkerMetadata CreateTestWorker(int id, string hostname = "localhost", int port = 5000)
    {
        return new WorkerMetadata
        {
            WorkerId = new WorkerId($"worker-{id}", hostname, port + id),
            Status = WorkerStatus.Joining,
            LocalWorldSize = 1
        };
    }

    [Test]
    public void RegisterWorker_WithValidMetadata_AddsWorkerToCluster()
    {
        // Arrange
        var worker = CreateTestWorker(1);
        var statusChangedCount = 0;
        WorkerStatus? lastStatus = null;

        _coordinator!.WorkerStatusChanged += (id, status) =>
        {
            statusChangedCount++;
            lastStatus = status;
        };

        // Act
        _coordinator.RegisterWorker(worker);

        // Assert
        var metadata = _coordinator.GetWorkerMetadata(worker.WorkerId);
        Assert.That(metadata, Is.Not.Null);
        Assert.That(metadata?.Status, Is.EqualTo(WorkerStatus.Joining));
        Assert.That(statusChangedCount, Is.EqualTo(1));
        Assert.That(lastStatus, Is.EqualTo(WorkerStatus.Joining));
    }

    [Test]
    public void RegisterWorker_WithDuplicateWorker_ThrowsInvalidOperationException()
    {
        // Arrange
        var worker = CreateTestWorker(1);
        _coordinator!.RegisterWorker(worker);

        // Act & Assert
        var ex = Assert.Throws<InvalidOperationException>(() => _coordinator.RegisterWorker(worker));
        Assert.That(ex.Message, Does.Contain("already registered"));
    }

    [Test]
    public void RegisterWorker_MultipleWorkers_AddsAllToCluster()
    {
        // Arrange
        var workers = new[]
        {
            CreateTestWorker(1),
            CreateTestWorker(2),
            CreateTestWorker(3)
        };

        // Act
        foreach (var worker in workers)
        {
            _coordinator!.RegisterWorker(worker);
        }

        // Assert
        var state = _coordinator!.GetClusterState();
        Assert.That(state.WorldSize, Is.EqualTo(0)); // Not yet stable

        // Wait for stability timer to fire
        Thread.Sleep(600);

        state = _coordinator.GetClusterState();
        Assert.That(state.WorldSize, Is.EqualTo(3));
    }

    [Test]
    public void UnregisterWorker_GracefulRemove_UpdatesStatusAndTopology()
    {
        // Arrange
        var worker = CreateTestWorker(1);
        _coordinator!.RegisterWorker(worker);

        WorkerStatus? lastStatus = null;
        _coordinator.WorkerStatusChanged += (id, status) => lastStatus = status;

        // Act
        _coordinator.UnregisterWorker(worker.WorkerId, graceful: true);

        // Assert
        var metadata = _coordinator.GetWorkerMetadata(worker.WorkerId);
        Assert.That(metadata?.Status, Is.EqualTo(WorkerStatus.Leaving));
        Assert.That(lastStatus, Is.EqualTo(WorkerStatus.Leaving));

        // Wait for rescaling trigger
        Thread.Sleep(600);

        var state = _coordinator.GetClusterState();
        Assert.That(state.WorldSize, Is.EqualTo(0));
    }

    [Test]
    public void UnregisterWorker_FailureRemove_UpdatesStatusAndIncrementsFailures()
    {
        // Arrange
        var worker = CreateTestWorker(1);
        _coordinator!.RegisterWorker(worker);

        WorkerStatus? lastStatus = null;
        _coordinator.WorkerStatusChanged += (id, status) => lastStatus = status;

        // Act
        _coordinator.UnregisterWorker(worker.WorkerId, graceful: false);

        // Assert
        var metadata = _coordinator.GetWorkerMetadata(worker.WorkerId);
        Assert.That(metadata?.Status, Is.EqualTo(WorkerStatus.Failed));
        Assert.That(lastStatus, Is.EqualTo(WorkerStatus.Failed));
    }

    [Test]
    public void UnregisterWorker_NonExistentWorker_DoesNothing()
    {
        // Arrange
        var nonExistentWorker = CreateTestWorker(999);

        // Act & Assert - Should not throw
        Assert.DoesNotThrow(() => _coordinator!.UnregisterWorker(nonExistentWorker.WorkerId));
    }

    [Test]
    public void UpdateHeartbeat_WithValidWorker_UpdatesTimestamp()
    {
        // Arrange
        var worker = CreateTestWorker(1);
        _coordinator!.RegisterWorker(worker);

        var oldTimestamp = worker.LastHeartbeat;
        Thread.Sleep(10);

        // Act
        _coordinator.UpdateHeartbeat(worker.WorkerId);

        // Assert
        var metadata = _coordinator.GetWorkerMetadata(worker.WorkerId);
        Assert.That(metadata?.LastHeartbeat, Is.GreaterThan(oldTimestamp));
    }

    [Test]
    public void UpdateHeartbeat_WithInvalidWorker_DoesNothing()
    {
        // Arrange
        var nonExistentWorker = CreateTestWorker(999);

        // Act & Assert - Should not throw
        Assert.DoesNotThrow(() => _coordinator!.UpdateHeartbeat(nonExistentWorker.WorkerId));
    }

    [Test]
    public void GetClusterState_ReturnsCorrectTopology()
    {
        // Arrange
        var workers = new[] { CreateTestWorker(1), CreateTestWorker(2) };
        foreach (var worker in workers)
        {
            _coordinator!.RegisterWorker(worker);
        }

        // Wait for stability
        Thread.Sleep(600);

        // Act
        var state = _coordinator!.GetClusterState();

        // Assert
        Assert.That(state.WorldSize, Is.EqualTo(2));
        Assert.That(state.Workers.Count, Is.EqualTo(2));
    }

    [Test]
    public void GetWorkerMetadata_WithValidWorker_ReturnsMetadata()
    {
        // Arrange
        var worker = CreateTestWorker(1);
        _coordinator!.RegisterWorker(worker);

        // Act
        var metadata = _coordinator.GetWorkerMetadata(worker.WorkerId);

        // Assert
        Assert.That(metadata, Is.Not.Null);
        Assert.That(metadata?.WorkerId, Is.EqualTo(worker.WorkerId));
    }

    [Test]
    public void GetWorkerMetadata_WithInvalidWorker_ReturnsNull()
    {
        // Arrange
        var nonExistentWorker = CreateTestWorker(999);

        // Act
        var metadata = _coordinator!.GetWorkerMetadata(nonExistentWorker.WorkerId);

        // Assert
        Assert.That(metadata, Is.Null);
    }

    [Test]
    public void GetWorkerMetadata_ReturnsClone_NotOriginal()
    {
        // Arrange
        var worker = CreateTestWorker(1);
        _coordinator!.RegisterWorker(worker);

        // Act
        var metadata1 = _coordinator.GetWorkerMetadata(worker.WorkerId);
        var metadata2 = _coordinator.GetWorkerMetadata(worker.WorkerId);

        // Assert - Should be different instances
        Assert.That(ReferenceEquals(metadata1, metadata2), Is.False);
    }

    [Test]
    public async Task BroadcastGlobalStateAsync_CompletesSuccessfully()
    {
        // Arrange
        var state = new GlobalTrainingState
        {
            CurrentEpoch = 5,
            CurrentStep = 100,
            LearningRate = 0.001f
        };

        // Act & Assert - Should not throw
        Assert.DoesNotThrowAsync(async () => await _coordinator!.BroadcastGlobalStateAsync(state));
    }

    [Test]
    public void TriggerRescaling_WithManualTrigger_FiresRescalingEvent()
    {
        // Arrange
        var worker = CreateTestWorker(1);
        _coordinator!.RegisterWorker(worker);
        Thread.Sleep(600);

        RescalingEvent? lastEvent = null;
        _coordinator.RescalingTriggered += evt => lastEvent = evt;

        // Act
        _coordinator.TriggerRescaling(RescaleType.ScaleUp);

        // Assert
        Assert.That(lastEvent, Is.Not.Null);
        Assert.That(lastEvent?.Type, Is.EqualTo(RescaleType.ScaleUp));
        Assert.That(lastEvent?.TriggerReason, Is.EqualTo("Manual trigger"));
    }

    [Test]
    public void GetGlobalState_ReturnsCorrectState()
    {
        // Arrange
        var worker = CreateTestWorker(1);
        _coordinator!.RegisterWorker(worker);
        Thread.Sleep(600);

        // Act
        var state = _coordinator.GetGlobalState(currentEpoch: 10, currentStep: 500, learningRate: 0.01f);

        // Assert
        Assert.That(state.CurrentEpoch, Is.EqualTo(10));
        Assert.That(state.CurrentStep, Is.EqualTo(500));
        Assert.That(state.LearningRate, Is.EqualTo(0.01f));
        Assert.That(state.ActiveWorkerCount, Is.EqualTo(1));
    }

    [Test]
    public void RescalingEvent_TriggeredAfterStabilityWindow()
    {
        // Arrange
        var rescalingEvents = new List<RescalingEvent>();
        _coordinator!.RescalingTriggered += evt => rescalingEvents.Add(evt);

        // Act
        var worker = CreateTestWorker(1);
        _coordinator.RegisterWorker(worker);

        // Wait for stability window
        Thread.Sleep(600);

        // Assert
        Assert.That(rescalingEvents.Count, Is.EqualTo(1));
        Assert.That(rescalingEvents[0].Type, Is.EqualTo(RescaleType.ScaleUp));
        Assert.That(rescalingEvents[0].AddedWorkers.Count, Is.EqualTo(1));
    }

    [Test]
    public void HealthCheck_DetectsFailedWorkers()
    {
        // Arrange
        var worker = CreateTestWorker(1);
        _coordinator!.RegisterWorker(worker);
        Thread.Sleep(600);

        WorkerStatus? lastStatus = null;
        _coordinator.WorkerStatusChanged += (id, status) => lastStatus = status;

        // Act - Simulate worker failure by not sending heartbeat
        // Wait longer than heartbeat timeout (1000ms) + health check interval (500ms)
        Thread.Sleep(1600);

        // Assert
        var metadata = _coordinator.GetWorkerMetadata(worker.WorkerId);
        // Worker should be marked as failed due to timeout
        Assert.That(lastStatus, Is.EqualTo(WorkerStatus.Failed));
    }

    [Test]
    public void UpdateHeartbeat_PreventsHealthCheckFailure()
    {
        // Arrange
        var worker = CreateTestWorker(1);
        _coordinator!.RegisterWorker(worker);
        Thread.Sleep(600);

        WorkerStatus? lastStatus = null;
        _coordinator.WorkerStatusChanged += (id, status) => lastStatus = status;

        // Act - Keep sending heartbeats
        for (int i = 0; i < 3; i++)
        {
            _coordinator.UpdateHeartbeat(worker.WorkerId);
            Thread.Sleep(400);
        }

        // Assert
        var metadata = _coordinator.GetWorkerMetadata(worker.WorkerId);
        Assert.That(metadata?.Status, Is.Not.EqualTo(WorkerStatus.Failed));
    }

    [Test]
    public void CheckFailureTolerance_WithConsecutiveFailures_ThrowsWhenExceeded()
    {
        // Arrange
        var config = new ElasticTrainingConfig
        {
            MinWorkers = 1,
            MaxWorkers = 8,
            MaxConsecutiveFailures = 2,
            FailureTolerancePercentage = 20
        };

        var coordinator = new ElasticCoordinator(config);
        var workers = new[] { CreateTestWorker(1), CreateTestWorker(2) };

        foreach (var worker in workers)
        {
            coordinator.RegisterWorker(worker);
        }
        Thread.Sleep(600);

        // Act & Assert
        // First failure - should not throw
        Assert.DoesNotThrow(() => coordinator.UnregisterWorker(workers[0].WorkerId, graceful: false));

        // Second failure - should not throw
        Assert.DoesNotThrow(() => coordinator.UnregisterWorker(workers[1].WorkerId, graceful: false));

        // Third failure - should throw
        var ex = Assert.Throws<InvalidOperationException>(() => coordinator.UnregisterWorker(workers[0].WorkerId, graceful: false));
        Assert.That(ex.Message, Does.Contain("Failure tolerance exceeded"));

        coordinator.Dispose();
    }

    [Test]
    public void CheckFailureTolerance_WithMinWorkersViolated_Throws()
    {
        // Arrange
        var config = new ElasticTrainingConfig
        {
            MinWorkers = 2,
            MaxWorkers = 8,
            MaxConsecutiveFailures = 10,
            FailureTolerancePercentage = 20
        };

        var coordinator = new ElasticCoordinator(config);
        var workers = new[] { CreateTestWorker(1), CreateTestWorker(2), CreateTestWorker(3) };

        foreach (var worker in workers)
        {
            coordinator.RegisterWorker(worker);
        }
        Thread.Sleep(600);

        // Act & Assert
        // Remove two workers - should still be ok
        Assert.DoesNotThrow(() => coordinator.UnregisterWorker(workers[0].WorkerId, graceful: false));
        Thread.Sleep(600);

        Assert.DoesNotThrow(() => coordinator.UnregisterWorker(workers[1].WorkerId, graceful: false));
        Thread.Sleep(600);

        // Try to remove the last worker below minimum - should throw
        var ex = Assert.Throws<InvalidOperationException>(() => coordinator.UnregisterWorker(workers[2].WorkerId, graceful: false));
        Assert.That(ex.Message, Does.Contain("Failure tolerance exceeded"));

        coordinator.Dispose();
    }

    [Test]
    public void ConcurrentRegisterWorker_ThreadSafeAccess()
    {
        // Arrange
        var workers = Enumerable.Range(1, 10).Select(i => CreateTestWorker(i)).ToArray();
        var exceptions = new List<Exception>();
        var lockObj = new object();

        // Act
        var tasks = workers.Select(worker => Task.Run(() =>
        {
            try
            {
                _coordinator!.RegisterWorker(worker);
            }
            catch (Exception ex)
            {
                lock (lockObj)
                {
                    exceptions.Add(ex);
                }
            }
        })).ToArray();

        Task.WaitAll(tasks);

        // Assert
        Assert.That(exceptions.Count, Is.EqualTo(0));

        // Wait for stability
        Thread.Sleep(600);

        var state = _coordinator!.GetClusterState();
        Assert.That(state.WorldSize, Is.EqualTo(10));
    }

    [Test]
    public void ConcurrentUnregisterWorker_ThreadSafeAccess()
    {
        // Arrange
        var workers = Enumerable.Range(1, 10).Select(i => CreateTestWorker(i)).ToArray();
        foreach (var worker in workers)
        {
            _coordinator!.RegisterWorker(worker);
        }
        Thread.Sleep(600);

        var exceptions = new List<Exception>();
        var lockObj = new object();

        // Act
        var tasks = workers.Take(5).Select(worker => Task.Run(() =>
        {
            try
            {
                _coordinator!.UnregisterWorker(worker.WorkerId);
            }
            catch (Exception ex)
            {
                lock (lockObj)
                {
                    exceptions.Add(ex);
                }
            }
        })).ToArray();

        Task.WaitAll(tasks);
        Thread.Sleep(600);

        // Assert
        Assert.That(exceptions.Count, Is.EqualTo(0));

        var state = _coordinator!.GetClusterState();
        Assert.That(state.WorldSize, Is.EqualTo(5));
    }

    [Test]
    public void WorkerStatusChanged_FiredForAllStatusTransitions()
    {
        // Arrange
        var statusChanges = new List<(WorkerId, WorkerStatus)>();
        _coordinator!.WorkerStatusChanged += (id, status) => statusChanges.Add((id, status));

        var worker = CreateTestWorker(1);

        // Act
        _coordinator.RegisterWorker(worker);
        Thread.Sleep(600);

        _coordinator.UnregisterWorker(worker.WorkerId, graceful: true);

        // Assert
        Assert.That(statusChanges.Count, Is.GreaterThanOrEqualTo(2));
        Assert.That(statusChanges.Any(sc => sc.Item2 == WorkerStatus.Joining), Is.True);
        Assert.That(statusChanges.Any(sc => sc.Item2 == WorkerStatus.Active), Is.True);
        Assert.That(statusChanges.Any(sc => sc.Item2 == WorkerStatus.Leaving), Is.True);
    }

    [Test]
    public void GetClusterState_ReturnsClone_NotOriginal()
    {
        // Arrange
        var worker = CreateTestWorker(1);
        _coordinator!.RegisterWorker(worker);
        Thread.Sleep(600);

        // Act
        var state1 = _coordinator.GetClusterState();
        var state2 = _coordinator.GetClusterState();

        // Assert - Should be different instances
        Assert.That(ReferenceEquals(state1, state2), Is.False);
    }
}
