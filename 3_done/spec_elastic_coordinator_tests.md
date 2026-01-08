# Spec: ElasticCoordinator Unit Tests

## Overview
Implement comprehensive unit tests for the ElasticCoordinator class to ensure correct cluster membership management, health monitoring, and rescaling trigger behavior.

## Deliverables

**File:** `tests/MachineLearning.Tests/Distributed/Coordinator/ElasticCoordinatorTests.cs`
```csharp
namespace MachineLearning.Tests.Distributed.Coordinator;

using MachineLearning.Distributed.Configuration;
using MachineLearning.Distributed.Coordinator;
using MachineLearning.Distributed.Enums;
using MachineLearning.Distributed.Models;

[TestFixture]
public class ElasticCoordinatorTests : IDisposable
{
    private ElasticTrainingConfig _config = null!;
    private ElasticCoordinator _coordinator = null!;
    private List<RescalingEvent> _capturedEvents = null!;

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
            FailureTolerancePercentage = 20
        };

        _coordinator = new ElasticCoordinator(_config);
        _capturedEvents = new List<RescalingEvent>();

        _coordinator.RescalingTriggered += evt => _capturedEvents.Add(evt);
        _coordinator.WorkerStatusChanged += (workerId, status) => { /* Capture status changes if needed */ };
    }

    [TearDown]
    public void TearDown()
    {
        _coordinator?.Dispose();
    }

    [Test]
    public void Constructor_ValidConfig_CreatesCoordinator()
    {
        // Assert
        Assert.That(_coordinator, Is.Not.Null);
    }

    [Test]
    public void Constructor_InvalidConfig_ThrowsArgumentException()
    {
        // Arrange
        var invalidConfig = new ElasticTrainingConfig
        {
            MinWorkers = -1,
            MaxWorkers = 10
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new ElasticCoordinator(invalidConfig));
    }

    [Test]
    public void RegisterWorker_NewWorker_AddsToCluster()
    {
        // Arrange
        var workerId = new WorkerId("worker1", "localhost", 8000);
        var metadata = CreateWorkerMetadata(workerId);

        // Act
        _coordinator.RegisterWorker(metadata);

        // Assert
        var state = _coordinator.GetClusterState();
        Assert.That(state.WorldSize, Is.EqualTo(1));
        Assert.That(state.Workers, Contains.Item(workerId));
    }

    [Test]
    public void RegisterWorker_DuplicateWorker_ThrowsException()
    {
        // Arrange
        var workerId = new WorkerId("worker1", "localhost", 8000);
        var metadata = CreateWorkerMetadata(workerId);
        _coordinator.RegisterWorker(metadata);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => _coordinator.RegisterWorker(metadata));
    }

    [Test]
    public void RegisterWorker_MultipleWorkers_AddsAllToCluster()
    {
        // Arrange
        var workers = new[]
        {
            new WorkerId("worker1", "localhost", 8000),
            new WorkerId("worker2", "localhost", 8001),
            new WorkerId("worker3", "localhost", 8002)
        };

        // Act
        foreach (var workerId in workers)
        {
            var metadata = CreateWorkerMetadata(workerId);
            _coordinator.RegisterWorker(metadata);
        }

        // Assert
        var state = _coordinator.GetClusterState();
        Assert.That(state.WorldSize, Is.EqualTo(3));
        Assert.That(state.Workers.Count, Is.EqualTo(3));
    }

    [Test]
    public void UnregisterWorker_GracefulRemoval_RemovesFromCluster()
    {
        // Arrange
        var workerId = new WorkerId("worker1", "localhost", 8000);
        var metadata = CreateWorkerMetadata(workerId);
        _coordinator.RegisterWorker(metadata);

        // Act
        _coordinator.UnregisterWorker(workerId, graceful: true);

        // Assert
        var state = _coordinator.GetClusterState();
        Assert.That(state.WorldSize, Is.EqualTo(0));
        Assert.That(state.Workers, Does.Not.Contain(workerId));
    }

    [Test]
    public void UnregisterWorker_NonExistentWorker_DoesNotThrow()
    {
        // Arrange
        var workerId = new WorkerId("worker1", "localhost", 8000);

        // Act & Assert
        Assert.DoesNotThrow(() => _coordinator.UnregisterWorker(workerId));
    }

    [Test]
    public void UpdateHeartbeat_ValidWorker_UpdatesTimestamp()
    {
        // Arrange
        var workerId = new WorkerId("worker1", "localhost", 8000);
        var metadata = CreateWorkerMetadata(workerId);
        _coordinator.RegisterWorker(metadata);

        var oldMetadata = _coordinator.GetWorkerMetadata(workerId);
        Thread.Sleep(10);

        // Act
        _coordinator.UpdateHeartbeat(workerId);
        var newMetadata = _coordinator.GetWorkerMetadata(workerId);

        // Assert
        Assert.That(newMetadata, Is.Not.Null);
        Assert.That(newMetadata.LastHeartbeat, Is.GreaterThan(oldMetadata.LastHeartbeat));
    }

    [Test]
    public void UpdateHeartbeat_NonExistentWorker_DoesNotThrow()
    {
        // Arrange
        var workerId = new WorkerId("worker1", "localhost", 8000);

        // Act & Assert
        Assert.DoesNotThrow(() => _coordinator.UpdateHeartbeat(workerId));
    }

    [Test]
    public void GetClusterState_ClusterEmpty_ReturnsEmptyState()
    {
        // Act
        var state = _coordinator.GetClusterState();

        // Assert
        Assert.That(state, Is.Not.Null);
        Assert.That(state.WorldSize, Is.EqualTo(0));
        Assert.That(state.Workers, Is.Empty);
    }

    [Test]
    public void GetWorkerMetadata_ExistingWorker_ReturnsMetadata()
    {
        // Arrange
        var workerId = new WorkerId("worker1", "localhost", 8000);
        var metadata = CreateWorkerMetadata(workerId);
        _coordinator.RegisterWorker(metadata);

        // Act
        var retrieved = _coordinator.GetWorkerMetadata(workerId);

        // Assert
        Assert.That(retrieved, Is.Not.Null);
        Assert.That(retrieved.WorkerId, Is.EqualTo(workerId));
    }

    [Test]
    public void GetWorkerMetadata_NonExistentWorker_ReturnsNull()
    {
        // Arrange
        var workerId = new WorkerId("worker1", "localhost", 8000);

        // Act
        var retrieved = _coordinator.GetWorkerMetadata(workerId);

        // Assert
        Assert.That(retrieved, Is.Null);
    }

    [Test]
    public void TriggerRescaling_ScaleUp_FiresEventWithCorrectType()
    {
        // Act
        _coordinator.TriggerRescaling(RescaleType.ScaleUp);

        // Assert
        Assert.That(_capturedEvents, Has.Count.EqualTo(1));
        Assert.That(_capturedEvents[0].Type, Is.EqualTo(RescaleType.ScaleUp));
    }

    [Test]
    public void TriggerRescaling_ScaleDown_FiresEventWithCorrectType()
    {
        // Act
        _coordinator.TriggerRescaling(RescaleType.ScaleDown);

        // Assert
        Assert.That(_capturedEvents, Has.Count.EqualTo(1));
        Assert.That(_capturedEvents[0].Type, Is.EqualTo(RescaleType.ScaleDown));
    }

    [Test]
    public void GetGlobalState_ReturnsCorrectState()
    {
        // Act
        var state = _coordinator.GetGlobalState(5, 100, 0.01f);

        // Assert
        Assert.That(state.CurrentEpoch, Is.EqualTo(5));
        Assert.That(state.CurrentStep, Is.EqualTo(100));
        Assert.That(state.LearningRate, Is.EqualTo(0.01f));
    }

    [Test]
    public void WorkerStatusChanged_RegisterWorker_FiresStatusChangedEvent()
    {
        // Arrange
        var workerId = new WorkerId("worker1", "localhost", 8000);
        var metadata = CreateWorkerMetadata(workerId);
        var statusChanged = false;
        WorkerStatus? capturedStatus = null;

        _coordinator.WorkerStatusChanged += (w, s) =>
        {
            if (w == workerId)
            {
                statusChanged = true;
                capturedStatus = s;
            }
        };

        // Act
        _coordinator.RegisterWorker(metadata);

        // Assert
        Assert.That(statusChanged, Is.True);
        Assert.That(capturedStatus, Is.EqualTo(WorkerStatus.Joining));
    }

    [Test]
    public void Dispose_ReleasesResources()
    {
        // Arrange
        var workerId = new WorkerId("worker1", "localhost", 8000);
        var metadata = CreateWorkerMetadata(workerId);
        _coordinator.RegisterWorker(metadata);

        // Act
        _coordinator.Dispose();

        // Assert - Should not throw when accessing state after dispose
        var state = _coordinator.GetClusterState();
        Assert.That(state, Is.Not.Null);
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

    public void Dispose()
    {
        _coordinator?.Dispose();
    }
}
```

## Test Coverage

The test suite should cover:

1. **Initialization**
   - Valid configuration creates coordinator
   - Invalid configuration throws exception

2. **Worker Registration**
   - Register single worker
   - Register multiple workers
   - Duplicate worker registration throws exception

3. **Worker Unregistration**
   - Graceful worker removal
   - Non-existent worker removal

4. **Heartbeat Management**
   - Update heartbeat updates timestamp
   - Non-existent worker heartbeat

5. **State Retrieval**
   - Get cluster state (empty and populated)
   - Get worker metadata (existing and non-existent)
   - Get global training state

6. **Rescaling Events**
   - Trigger scale up event
   - Trigger scale down event
   - Event contains correct topology information

7. **Worker Status Events**
   - Status changed fires on registration
   - Status changed fires on unregistration

8. **Resource Management**
   - Dispose releases resources correctly

## Implementation Notes

1. Use NUnit testing framework
2. Setup/Teardown methods for clean test isolation
3. Helper methods for creating test data
4. Async tests for async methods
5. Mock external dependencies if needed

## Dependencies
- NUnit testing framework
- ElasticCoordinator implementation from spec_elastic_coordinator_core.md
- Configuration and models from spec_elastic_config_models.md

## Estimated Effort
~45 minutes

## Success Criteria
- All tests pass
- Test coverage > 90% for ElasticCoordinator
- Tests are clear, maintainable, and well-documented
- Edge cases are properly tested
