# Spec: ElasticWorker Unit Tests

## Overview
Implement comprehensive unit tests for the ElasticWorker class to ensure correct connection handling, rescaling event processing, and state management.

## Deliverables

**File:** `tests/MachineLearning.Tests/Distributed/Worker/ElasticWorkerTests.cs`
```csharp
namespace MachineLearning.Tests.Distributed.Worker;

using MachineLearning.Distributed.Configuration;
using MachineLearning.Distributed.Models;
using MachineLearning.Distributed.Worker;

[TestFixture]
public class ElasticWorkerTests : IDisposable
{
    private ElasticTrainingConfig _config = null!;
    private WorkerId _workerId = null!;
    private ElasticWorker _worker = null!;
    private List<RescalingEvent> _capturedEvents = null!;
    private List<GlobalTrainingState> _synchronizedStates = null!;

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
            UseSynchronousRescaling = true
        };

        _workerId = new WorkerId("worker1", "localhost", 8000);
        _worker = new ElasticWorker(_workerId, _config);

        _capturedEvents = new List<RescalingEvent>();
        _synchronizedStates = new List<GlobalTrainingState>();

        _worker.RescalingRequested += evt => _capturedEvents.Add(evt);
        _worker.StateSynchronized += state => _synchronizedStates.Add(state);
    }

    [TearDown]
    public void TearDown()
    {
        _worker?.Dispose();
    }

    [Test]
    public void Constructor_ValidParameters_CreatesWorker()
    {
        // Assert
        Assert.That(_worker, Is.Not.Null);
        Assert.That(_worker.WorkerId, Is.EqualTo(_workerId));
        Assert.That(_worker.CurrentTopology, Is.Null);
        Assert.That(_worker.CurrentState, Is.Null);
        Assert.That(_worker.IsRescaling, Is.False);
    }

    [Test]
    public void ConnectToCluster_ValidAddress_ConnectsSuccessfully()
    {
        // Act
        _worker.ConnectToCluster("localhost:9000");

        // Assert
        var metadata = _worker.GetMetadata();
        Assert.That(metadata.Status, Is.EqualTo(WorkerStatus.Joining));
    }

    [Test]
    public void ConnectToCluster_AlreadyConnected_ThrowsException()
    {
        // Arrange
        _worker.ConnectToCluster("localhost:9000");

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => _worker.ConnectToCluster("localhost:9000"));
    }

    [Test]
    public async Task DisconnectAsync_ConnectedWorker_DisconnectsSuccessfully()
    {
        // Arrange
        _worker.ConnectToCluster("localhost:9000");

        // Act
        await _worker.DisconnectAsync();

        // Assert
        var metadata = _worker.GetMetadata();
        Assert.That(metadata.Status, Is.EqualTo(WorkerStatus.Leaving));
    }

    [Test]
    public async Task DisconnectAsync_NotConnected_DoesNotThrow()
    {
        // Act & Assert
        Assert.DoesNotThrowAsync(async () => await _worker.DisconnectAsync());
    }

    [Test]
    public async Task OnRescalingEventAsync_ScaleUp_UpdatesStateAndFiresEvent()
    {
        // Arrange
        _worker.ConnectToCluster("localhost:9000");
        var oldTopology = new ClusterTopology { WorldSize = 2, Workers = new List<WorkerId> { _workerId } };
        var newTopology = new ClusterTopology { WorldSize = 4, Workers = new List<WorkerId> { _workerId } };

        var evt = new RescalingEvent
        {
            Type = RescaleType.ScaleUp,
            OldTopology = oldTopology,
            NewTopology = newTopology,
            AddedWorkers = new List<WorkerId>(),
            RemovedWorkers = new List<WorkerId>()
        };

        // Act
        await _worker.OnRescalingEventAsync(evt);

        // Assert
        Assert.That(_capturedEvents, Has.Count.EqualTo(1));
        Assert.That(_capturedEvents[0].Type, Is.EqualTo(RescaleType.ScaleUp));
        Assert.That(_worker.IsRescaling, Is.False);
    }

    [Test]
    public async Task OnRescalingEventAsync_ScaleDown_UpdatesStateAndFiresEvent()
    {
        // Arrange
        _worker.ConnectToCluster("localhost:9000");
        var oldTopology = new ClusterTopology { WorldSize = 4, Workers = new List<WorkerId> { _workerId } };
        var newTopology = new ClusterTopology { WorldSize = 2, Workers = new List<WorkerId> { _workerId } };

        var evt = new RescalingEvent
        {
            Type = RescaleType.ScaleDown,
            OldTopology = oldTopology,
            NewTopology = newTopology,
            AddedWorkers = new List<WorkerId>(),
            RemovedWorkers = new List<WorkerId>()
        };

        // Act
        await _worker.OnRescalingEventAsync(evt);

        // Assert
        Assert.That(_capturedEvents, Has.Count.EqualTo(1));
        Assert.That(_capturedEvents[0].Type, Is.EqualTo(RescaleType.ScaleDown));
    }

    [Test]
    public async Task UpdateTopologyAsync_ValidTopology_UpdatesWorkerTopology()
    {
        // Arrange
        var topology = new ClusterTopology
        {
            WorldSize = 2,
            Workers = new List<WorkerId> { _workerId }
        };

        // Act
        await _worker.UpdateTopologyAsync(topology);

        // Assert
        Assert.That(_worker.CurrentTopology, Is.EqualTo(topology));
        var metadata = _worker.GetMetadata();
        Assert.That(metadata.Rank, Is.EqualTo(0));
    }

    [Test]
    public async Task UpdateTopologyAsync_WorkerNotInTopology_SetsRankToMinusOne()
    {
        // Arrange
        var otherWorker = new WorkerId("worker2", "localhost", 8001);
        var topology = new ClusterTopology
        {
            WorldSize = 1,
            Workers = new List<WorkerId> { otherWorker }
        };

        // Act
        await _worker.UpdateTopologyAsync(topology);

        // Assert
        var metadata = _worker.GetMetadata();
        Assert.That(metadata.Rank, Is.EqualTo(-1));
    }

    [Test]
    public async Task SynchronizeStateAsync_ValidState_UpdatesWorkerState()
    {
        // Arrange
        var state = new GlobalTrainingState
        {
            CurrentEpoch = 5,
            CurrentStep = 1000,
            LearningRate = 0.01f,
            ActiveWorkerCount = 4
        };

        // Act
        await _worker.SynchronizeStateAsync(state);

        // Assert
        Assert.That(_worker.CurrentState, Is.EqualTo(state));
        Assert.That(_synchronizedStates, Has.Count.EqualTo(1));
        Assert.That(_synchronizedStates[0], Is.EqualTo(state));
    }

    [Test]
    public async Task SynchronizeStateAsync_FiresStateSynchronizedEvent()
    {
        // Arrange
        var state = new GlobalTrainingState
        {
            CurrentEpoch = 5,
            CurrentStep = 1000,
            LearningRate = 0.01f,
            ActiveWorkerCount = 4
        };

        // Act
        await _worker.SynchronizeStateAsync(state);

        // Assert
        Assert.That(_synchronizedStates, Has.Count.EqualTo(1));
    }

    [Test]
    public async Task ResumeTrainingAsync_WhileRescaling_ResetsRescalingFlag()
    {
        // Arrange
        _worker.ConnectToCluster("localhost:9000");
        var evt = new RescalingEvent
        {
            Type = RescaleType.ScaleUp,
            OldTopology = new ClusterTopology(),
            NewTopology = new ClusterTopology()
        };

        await _worker.OnRescalingEventAsync(evt);

        // Act
        await _worker.ResumeTrainingAsync();

        // Assert
        var metadata = _worker.GetMetadata();
        Assert.That(metadata.Status, Is.EqualTo(WorkerStatus.Active));
    }

    [Test]
    public async Task RedistributeDataAsync_ValidPlan_DoesNotThrow()
    {
        // Arrange
        var plan = new DataRedistributionPlan
        {
            Transfers = new List<DataTransfer>(),
            WorkerAssignments = new Dictionary<WorkerId, List<DataShard>>(),
            TotalShards = 100
        };

        // Act & Assert
        Assert.DoesNotThrowAsync(async () => await _worker.RedistributeDataAsync(plan));
    }

    [Test]
    public void UpdateTrainingState_ValidParameters_UpdatesCurrentState()
    {
        // Arrange
        var initialState = new GlobalTrainingState
        {
            CurrentEpoch = 1,
            CurrentStep = 100,
            LearningRate = 0.01f,
            ActiveWorkerCount = 4
        };

        // Manually set state (simulating synchronization)
        _worker.GetType().GetProperty("CurrentState")?
            .SetValue(_worker, initialState);

        // Act
        _worker.UpdateTrainingState(2, 200, 0.005f);

        // Assert
        Assert.That(_worker.CurrentState.CurrentEpoch, Is.EqualTo(2));
        Assert.That(_worker.CurrentState.CurrentStep, Is.EqualTo(200));
        Assert.That(_worker.CurrentState.LearningRate, Is.EqualTo(0.005f));
    }

    [Test]
    public void ShouldHandleShard_WorkerInTopology_ReturnsCorrectResult()
    {
        // Arrange
        var topology = new ClusterTopology
        {
            WorldSize = 4,
            Workers = new List<WorkerId> { _workerId }
        };

        // Manually set topology and rank
        _worker.GetType().GetProperty("CurrentTopology")?.SetValue(_worker, topology);
        var metadata = _worker.GetMetadata();
        metadata.Rank = 0;

        // Act
        var shouldHandle = _worker.ShouldHandleShard(0, 10);

        // Assert
        Assert.That(shouldHandle, Is.True);
    }

    [Test]
    public void ShouldHandleShard_WorkerNotAssignedShard_ReturnsFalse()
    {
        // Arrange
        var topology = new ClusterTopology
        {
            WorldSize = 4,
            Workers = new List<WorkerId> { _workerId }
        };

        // Manually set topology and rank
        _worker.GetType().GetProperty("CurrentTopology")?.SetValue(_worker, topology);
        var metadata = _worker.GetMetadata();
        metadata.Rank = 0;

        // Act
        var shouldHandle = _worker.ShouldHandleShard(3, 10);

        // Assert
        Assert.That(shouldHandle, Is.False);
    }

    [Test]
    public void GetMetadata_ReturnsCorrectWorkerMetadata()
    {
        // Act
        var metadata = _worker.GetMetadata();

        // Assert
        Assert.That(metadata, Is.Not.Null);
        Assert.That(metadata.WorkerId, Is.EqualTo(_workerId));
        Assert.That(metadata.Status, Is.EqualTo(WorkerStatus.Joining));
        Assert.That(metadata.Rank, Is.EqualTo(-1));
    }

    [Test]
    public void Dispose_StopsHeartbeatAndReleasesResources()
    {
        // Arrange
        _worker.ConnectToCluster("localhost:9000");

        // Act
        _worker.Dispose();

        // Assert - Should not throw
        var metadata = _worker.GetMetadata();
        Assert.That(metadata, Is.Not.Null);
    }

    #region Helper Methods

    private RescalingEvent CreateRescalingEvent(RescaleType type)
    {
        return new RescalingEvent
        {
            Type = type,
            OldTopology = new ClusterTopology(),
            NewTopology = new ClusterTopology(),
            AddedWorkers = new List<WorkerId>(),
            RemovedWorkers = new List<WorkerId>()
        };
    }

    #endregion

    public void Dispose()
    {
        _worker?.Dispose();
    }
}
```

## Test Coverage

The test suite should cover:

1. **Initialization**
   - Valid parameters create worker
   - Worker properties are initialized correctly

2. **Connection Management**
   - Connect to coordinator successfully
   - Duplicate connection throws exception
   - Disconnect gracefully
   - Disconnect when not connected

3. **Rescaling Events**
   - Handle scale up events
   - Handle scale down events
   - Update topology correctly
   - Fire appropriate events

4. **State Management**
   - Update topology correctly
   - Synchronize training state
   - Update training state parameters
   - Handle worker not in topology

5. **Data Distribution**
   - Redistribute data with valid plan
   - Determine shard assignment correctly

6. **Event Handling**
   - RescalingRequested event fires
   - StateSynchronized event fires

7. **Resource Management**
   - Dispose releases resources correctly

## Implementation Notes

1. Use NUnit testing framework
2. Setup/Teardown methods for clean test isolation
3. Helper methods for creating test data
4. Async tests for async methods
5. Use reflection to set private properties when needed for testing

## Dependencies
- NUnit testing framework
- ElasticWorker implementation from spec_elastic_worker_core.md
- Configuration and models from spec_elastic_config_models.md

## Estimated Effort
~50 minutes

## Success Criteria
- All tests pass
- Test coverage > 90% for ElasticWorker
- Tests are clear, maintainable, and well-documented
- Edge cases are properly tested
- Async operations are properly tested
