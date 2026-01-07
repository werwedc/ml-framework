# Spec: Communication Unit Tests

## Overview
Comprehensive unit tests for all communication primitives, including backend mocks, operation tests, and integration tests.

## Dependencies
- `spec_communication_interfaces.md`
- `spec_process_group.md`
- `spec_collective_basic.md`
- `spec_collective_advanced.md`
- `spec_point_to_point.md`
- `spec_async_operations.md`
- All backend specs

## Technical Requirements

### 1. Mock Communication Backend
Mock implementation for testing without real hardware.

```csharp
namespace MLFramework.Tests.Communication
{
    /// <summary>
    /// Mock communication backend for testing
    /// </summary>
    public class MockCommunicationBackend : IAsyncCommunicationBackend, IPointToPointCommunication
    {
        private readonly int _rank;
        private readonly int _worldSize;
        private readonly Dictionary<int, Dictionary<int, Tensor<float>>> _dataStore;
        private readonly List<string> _operationLog;
        private readonly Dictionary<int, Queue<MessageInfo>> _messageQueues;
        private readonly Dictionary<int, List<MessageInfo>> _probeResults;
        private bool _delayEnabled;
        private int _delayMs;
        private bool _shouldFail;
        private int _failureCount;

        public int Rank => _rank;
        public int WorldSize => _worldSize;
        public string BackendName => "Mock";
        public IReadOnlyList<string> OperationLog => _operationLog.AsReadOnly();

        public MockCommunicationBackend(int rank, int worldSize)
        {
            _rank = rank;
            _worldSize = worldSize;
            _dataStore = new Dictionary<int, Dictionary<int, Tensor<float>>>();
            _operationLog = new List<string>();
            _messageQueues = new Dictionary<int, Queue<MessageInfo>>();
            _probeResults = new Dictionary<int, List<MessageInfo>>();

            // Initialize data store for each rank
            for (int r = 0; r < worldSize; r++)
            {
                _dataStore[r] = new Dictionary<int, Tensor<float>>();
                _messageQueues[r] = new Queue<MessageInfo>();
            }
        }

        public void Broadcast<T>(Tensor<T> tensor, int rootRank)
        {
            LogOperation("Broadcast", rootRank);
            SimulateDelay();
            SimulateFailure();

            // Store tensor on all ranks
            for (int r = 0; r < _worldSize; r++)
            {
                _dataStore[r][-1] = tensor.Clone() as Tensor<T>;
            }
        }

        public Tensor<T> Reduce<T>(Tensor<T> tensor, ReduceOp operation, int rootRank)
        {
            LogOperation("Reduce", rootRank);
            SimulateDelay();
            SimulateFailure();

            // For simplicity, just return the tensor as-is
            if (_rank == rootRank)
            {
                return tensor.Clone() as Tensor<T>;
            }
            return null;
        }

        public Tensor<T> AllReduce<T>(Tensor<T> tensor, ReduceOp operation)
        {
            LogOperation("AllReduce", _rank);
            SimulateDelay();
            SimulateFailure();

            // For simplicity, just return the tensor as-is
            return tensor.Clone() as Tensor<T>;
        }

        public Tensor<T> AllGather<T>(Tensor<T> tensor)
        {
            LogOperation("AllGather", _rank);
            SimulateDelay();
            SimulateFailure();

            // Gather data from all ranks
            var gatheredData = new List<Tensor<T>>();
            for (int r = 0; r < _worldSize; r++)
            {
                gatheredData.Add(tensor.Clone() as Tensor<T>);
            }

            // Concatenate (simplified)
            return tensor.Clone() as Tensor<T>;
        }

        public Tensor<T> ReduceScatter<T>(Tensor<T> tensor, ReduceOp operation)
        {
            LogOperation("ReduceScatter", _rank);
            SimulateDelay();
            SimulateFailure();

            // Return a slice of the tensor
            return tensor.Clone() as Tensor<T>;
        }

        public void Barrier()
        {
            LogOperation("Barrier", _rank);
            SimulateDelay();
            SimulateFailure();
        }

        // Point-to-point operations
        public void Send<T>(Tensor<T> tensor, int destinationRank, int tag = 0)
        {
            LogOperation($"Send to rank {destinationRank}", _rank);
            SimulateDelay();
            SimulateFailure();

            _dataStore[destinationRank][tag] = tensor.Clone() as Tensor<T>;

            // Add to message queue
            _messageQueues[destinationRank].Enqueue(new MessageInfo
            {
                SourceRank = _rank,
                Tag = tag,
                Count = tensor.Shape.TotalSize,
                DataType = typeof(T)
            });
        }

        public Tensor<T> Receive<T>(int sourceRank, int tag = 0)
        {
            LogOperation($"Receive from rank {sourceRank}", _rank);
            SimulateDelay();
            SimulateFailure();

            if (_dataStore[_rank].TryGetValue(tag, out var tensor))
            {
                _dataStore[_rank].Remove(tag);
                return tensor.Clone() as Tensor<T>;
            }

            throw new CommunicationException($"No message available from rank {sourceRank}");
        }

        public Tensor<T> Receive<T>(int sourceRank, Tensor<T> template, int tag = 0)
        {
            return Receive<T>(sourceRank, tag);
        }

        public MessageInfo? Probe(int sourceRank, int tag = 0)
        {
            LogOperation($"Probe rank {sourceRank}", _rank);
            SimulateDelay();

            if (_probeResults.TryGetValue(_rank, out var probes))
            {
                return probes.FirstOrDefault(p => p.SourceRank == sourceRank && p.Tag == tag);
            }

            return _messageQueues[_rank].FirstOrDefault(m => m.Tag == tag);
        }

        // Async operations
        public ICommunicationHandle BroadcastAsync<T>(Tensor<T> tensor, int rootRank)
        {
            var task = Task.Run(() => Broadcast(tensor, rootRank));
            return new AsyncCommunicationHandle(task);
        }

        public ICommunicationHandle AllReduceAsync<T>(Tensor<T> tensor, ReduceOp operation)
        {
            var task = Task.Run(() => AllReduce(tensor, operation));
            return new AsyncCommunicationHandle(task);
        }

        public ICommunicationHandle BarrierAsync()
        {
            var task = Task.Run(() => Barrier());
            return new AsyncCommunicationHandle(task);
        }

        public ICommunicationHandle SendAsync<T>(Tensor<T> tensor, int destinationRank, int tag = 0)
        {
            var task = Task.Run(() => Send(tensor, destinationRank, tag));
            return new AsyncCommunicationHandle(task);
        }

        public ICommunicationHandle ReceiveAsync<T>(int sourceRank, int tag = 0)
        {
            var task = Task.Run(() => Receive<T>(sourceRank, tag));
            return new AsyncCommunicationHandle(task);
        }

        public ICommunicationHandle ReceiveAsync<T>(int sourceRank, Tensor<T> template, int tag = 0)
        {
            return ReceiveAsync<T>(sourceRank, tag);
        }

        // Configuration methods
        public void SetDelay(int delayMs)
        {
            _delayEnabled = true;
            _delayMs = delayMs;
        }

        public void DisableDelay()
        {
            _delayEnabled = false;
        }

        public void SetFailure(bool shouldFail, int count = 1)
        {
            _shouldFail = shouldFail;
            _failureCount = count;
        }

        public void AddProbeResult(int rank, MessageInfo info)
        {
            if (!_probeResults.ContainsKey(rank))
            {
                _probeResults[rank] = new List<MessageInfo>();
            }
            _probeResults[rank].Add(info);
        }

        private void LogOperation(string operation, int rank)
        {
            _operationLog.Add($"Rank {rank}: {operation} at {DateTime.Now:HH:mm:ss.fff}");
        }

        private void SimulateDelay()
        {
            if (_delayEnabled && _delayMs > 0)
            {
                Thread.Sleep(_delayMs);
            }
        }

        private void SimulateFailure()
        {
            if (_shouldFail && _failureCount > 0)
            {
                _failureCount--;
                throw new CommunicationException("Simulated failure");
            }
        }

        public void Dispose()
        {
            // Cleanup
        }
    }
}
```

### 2. Interface Tests
Tests for communication interfaces and exceptions.

```csharp
namespace MLFramework.Tests.Communication
{
    [TestFixture]
    public class CommunicationInterfaceTests
    {
        [Test]
        public void TestReduceOp_Values()
        {
            // Test that ReduceOp enum has expected values
            Assert.AreEqual(0, (int)ReduceOp.Sum);
            Assert.AreEqual(1, (int)ReduceOp.Product);
            Assert.AreEqual(2, (int)ReduceOp.Max);
            Assert.AreEqual(3, (int)ReduceOp.Min);
            Assert.AreEqual(4, (int)ReduceOp.Avg);
        }

        [Test]
        public void TestDeviceType_Values()
        {
            // Test that DeviceType enum has expected values
            Assert.AreEqual(0, (int)DeviceType.CPU);
            Assert.AreEqual(1, (int)DeviceType.CUDA);
            Assert.AreEqual(2, (int)DeviceType.ROCm);
        }

        [Test]
        public void TestCommunicationException_Message()
        {
            var ex = new CommunicationException("Test message");
            Assert.AreEqual("Test message", ex.Message);
            Assert.IsNull(ex.Rank);
            Assert.IsNull(ex.BackendName);
        }

        [Test]
        public void TestCommunicationException_WithRankAndBackend()
        {
            var ex = new CommunicationException("Test message", 0, "NCCL");
            Assert.AreEqual("Test message", ex.Message);
            Assert.AreEqual(0, ex.Rank);
            Assert.AreEqual("NCCL", ex.BackendName);
        }

        [Test]
        public void TestCommunicationException_WithInnerException()
        {
            var inner = new Exception("Inner");
            var ex = new CommunicationException("Test message", inner);
            Assert.AreEqual("Test message", ex.Message);
            Assert.AreEqual(inner, ex.InnerException);
        }

        [Test]
        public void TestCommunicationTimeoutException()
        {
            var ex = new CommunicationTimeoutException("Timeout", TimeSpan.FromSeconds(5));
            Assert.AreEqual("Timeout", ex.Message);
            Assert.AreEqual(TimeSpan.FromSeconds(5), ex.TimeoutDuration);
        }

        [Test]
        public void TestRankMismatchException()
        {
            var ex = new RankMismatchException("Mismatch", 0, 1);
            Assert.AreEqual("Mismatch", ex.Message);
            Assert.AreEqual(0, ex.ExpectedRank);
            Assert.AreEqual(1, ex.ActualRank);
        }

        [Test]
        public void TestCommunicationConfig_DefaultValues()
        {
            var config = new CommunicationConfig();
            Assert.AreEqual(300000, config.TimeoutMs);
            Assert.IsFalse(config.EnableLogging);
            Assert.IsTrue(config.UsePinnedMemory);
        }

        [Test]
        public void TestCommunicationConfig_CustomValues()
        {
            var config = new CommunicationConfig
            {
                TimeoutMs = 60000,
                EnableLogging = true,
                UsePinnedMemory = false
            };
            Assert.AreEqual(60000, config.TimeoutMs);
            Assert.IsTrue(config.EnableLogging);
            Assert.IsFalse(config.UsePinnedMemory);
        }
    }
}
```

### 3. Collective Operation Tests
Tests for collective operations.

```csharp
namespace MLFramework.Tests.Communication
{
    [TestFixture]
    public class CollectiveOperationTests
    {
        private MockCommunicationBackend _backend;
        private Tensor<float> _tensor;

        [SetUp]
        public void Setup()
        {
            _backend = new MockCommunicationBackend(0, 4);
            _tensor = CreateTestTensor(10);
        }

        [TearDown]
        public void TearDown()
        {
            _backend?.Dispose();
            _tensor?.Dispose();
        }

        [Test]
        public void TestBroadcast_Success()
        {
            // Arrange
            var result = _tensor.Clone();

            // Act
            _backend.Broadcast(result, 0);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(10, result.Shape.TotalSize);
        }

        [Test]
        public void TestBroadcast_InvalidRootRank_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                _backend.Broadcast(_tensor, 10));
        }

        [Test]
        public void TestAllReduce_Sum()
        {
            // Act
            var result = _backend.AllReduce(_tensor, ReduceOp.Sum);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(10, result.Shape.TotalSize);
        }

        [Test]
        public void TestAllReduce_AllReduceOperations()
        {
            // Test all reduce operations
            var operations = new[] { ReduceOp.Sum, ReduceOp.Product, ReduceOp.Max, ReduceOp.Min };

            foreach (var op in operations)
            {
                var result = _backend.AllReduce(_tensor, op);
                Assert.IsNotNull(result, $"AllReduce with {op} failed");
            }
        }

        [Test]
        public void TestAllGather_Success()
        {
            // Act
            var result = _backend.AllGather(_tensor);

            // Assert
            Assert.IsNotNull(result);
        }

        [Test]
        public void TestReduceScatter_Success()
        {
            // Act
            var result = _backend.ReduceScatter(_tensor, ReduceOp.Sum);

            // Assert
            Assert.IsNotNull(result);
        }

        [Test]
        public void TestBarrier_Success()
        {
            // Act & Assert - should not throw
            _backend.Barrier();
        }

        [Test]
        public void TestAsyncAllReduce_CompletesSuccessfully()
        {
            // Act
            var handle = _backend.AllReduceAsync(_tensor, ReduceOp.Sum);
            handle.Wait();

            // Assert
            Assert.IsTrue(handle.IsCompleted);
            Assert.IsNotNull(handle.GetResult<float>());
        }

        [Test]
        public void TestAsyncAllReduce_TryWait_ReturnsTrue()
        {
            // Act
            var handle = _backend.AllReduceAsync(_tensor, ReduceOp.Sum);
            var completed = handle.TryWait(5000);

            // Assert
            Assert.IsTrue(completed);
            Assert.IsTrue(handle.IsCompleted);
        }

        [Test]
        public void TestAsyncAllReduce_TryGetResultBeforeComplete_ThrowsException()
        {
            // Arrange
            _backend.SetDelay(100);
            var handle = _backend.AllReduceAsync(_tensor, ReduceOp.Sum);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => handle.GetResult<float>());
        }

        private Tensor<float> CreateTestTensor(int size)
        {
            var data = Enumerable.Range(0, size).Select(i => (float)i).ToArray();
            return new Tensor<float>(data);
        }
    }
}
```

### 4. Point-to-Point Tests
Tests for point-to-point operations.

```csharp
namespace MLFramework.Tests.Communication
{
    [TestFixture]
    public class PointToPointTests
    {
        private MockCommunicationBackend _backend;
        private MockCommunicationBackend _remoteBackend;
        private Tensor<float> _tensor;

        [SetUp]
        public void Setup()
        {
            _backend = new MockCommunicationBackend(0, 2);
            _remoteBackend = new MockCommunicationBackend(1, 2);
            _tensor = CreateTestTensor(10);
        }

        [TearDown]
        public void TearDown()
        {
            _backend?.Dispose();
            _remoteBackend?.Dispose();
            _tensor?.Dispose();
        }

        [Test]
        public void TestSend_InvalidDestinationRank_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _backend.Send(_tensor, _backend.Rank)); // Send to self
        }

        [Test]
        public void TestSendAndReceive_Success()
        {
            // Arrange
            var received = _remoteBackend.Receive<float>(_backend.Rank, 0);

            // Act
            _backend.Send(_tensor, _remoteBackend.Rank, 0);

            // Assert
            // Note: This is a simplified test - in reality, we'd need proper synchronization
            Assert.IsNotNull(received);
        }

        [Test]
        public void TestReceive_InvalidSourceRank_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _backend.Receive<float>(_backend.Rank)); // Receive from self
        }

        [Test]
        public void TestProbe_NoMessage_ReturnsNull()
        {
            // Act
            var result = _backend.Probe(1, 0);

            // Assert
            Assert.IsNull(result);
        }

        [Test]
        public void TestProbe_WithMessage_ReturnsMessageInfo()
        {
            // Arrange
            _backend.AddProbeResult(0, new MessageInfo
            {
                SourceRank = 1,
                Tag = 0,
                Count = 10,
                DataType = typeof(float)
            });

            // Act
            var result = _backend.Probe(1, 0);

            // Assert
            Assert.IsNotNull(result);
            Assert.AreEqual(1, result.SourceRank);
            Assert.AreEqual(0, result.Tag);
            Assert.AreEqual(10, result.Count);
        }

        [Test]
        public void TestSendAsync_CompletesSuccessfully()
        {
            // Act
            var handle = _backend.SendAsync(_tensor, _remoteBackend.Rank, 0);
            handle.Wait();

            // Assert
            Assert.IsTrue(handle.IsCompleted);
        }

        [Test]
        public void TestReceiveAsync_CompletesSuccessfully()
        {
            // Arrange
            var handle = _remoteBackend.ReceiveAsync<float>(_backend.Rank, 0);

            // Act
            _backend.Send(_tensor, _remoteBackend.Rank, 0);
            handle.Wait();

            // Assert
            Assert.IsTrue(handle.IsCompleted);
        }

        private Tensor<float> CreateTestTensor(int size)
        {
            var data = Enumerable.Range(0, size).Select(i => (float)i).ToArray();
            return new Tensor<float>(data);
        }
    }
}
```

### 5. Process Group Tests
Tests for process group management.

```csharp
namespace MLFramework.Tests.Communication
{
    [TestFixture]
    public class ProcessGroupTests
    {
        private MockCommunicationBackend _backend;
        private ProcessGroupManager _manager;

        [SetUp]
        public void Setup()
        {
            _backend = new MockCommunicationBackend(0, 8);
            _manager = new ProcessGroupManager(_backend);
        }

        [TearDown]
        public void TearDown()
        {
            _manager?.Dispose();
            _backend?.Dispose();
        }

        [Test]
        public void TestProcessGroupManager_WorldGroupExists()
        {
            // Act
            var worldGroup = _manager.WorldGroup;

            // Assert
            Assert.IsNotNull(worldGroup);
            Assert.AreEqual("world", worldGroup.GroupName);
            Assert.AreEqual(8, worldGroup.GroupSize);
            Assert.IsTrue(worldGroup.IsWorldGroup);
        }

        [Test]
        public void TestCreateGroup_Success()
        {
            // Act
            var group = _manager.CreateGroup("test", new[] { 0, 1, 2, 3 });

            // Assert
            Assert.IsNotNull(group);
            Assert.AreEqual("test", group.GroupName);
            Assert.AreEqual(4, group.GroupSize);
        }

        [Test]
        public void TestCreateGroup_DuplicateName_ThrowsException()
        {
            // Arrange
            _manager.CreateGroup("test", new[] { 0, 1 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _manager.CreateGroup("test", new[] { 2, 3 }));
        }

        [Test]
        public void TestCreateGroup_RankNotInWorld_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _manager.CreateGroup("test", new[] { 10, 11 })); // Invalid ranks
        }

        [Test]
        public void TestDestroyGroup_Success()
        {
            // Arrange
            _manager.CreateGroup("test", new[] { 0, 1 });

            // Act & Assert - should not throw
            _manager.DestroyGroup("test");
        }

        [Test]
        public void TestDestroyWorldGroup_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                _manager.DestroyGroup("world"));
        }

        [Test]
        public void TestCreateSubGroup_Success()
        {
            // Arrange
            var parentGroup = _manager.CreateGroup("parent", new[] { 0, 1, 2, 3 });

            // Act
            var subGroup = parentGroup.CreateSubGroup("child", new[] { 0, 1 });

            // Assert
            Assert.IsNotNull(subGroup);
            Assert.AreEqual("child", subGroup.GroupName);
            Assert.AreEqual(2, subGroup.GroupSize);
        }

        [Test]
        public void TestProcessGroupHelpers_CreateSplitGroups()
        {
            // Act
            var groups = ProcessGroupHelpers.CreateSplitGroups(_manager, 4);

            // Assert
            Assert.AreEqual(4, groups.Count);
            foreach (var kvp in groups)
            {
                Assert.AreEqual(2, kvp.Value.GroupSize);
            }
        }

        [Test]
        public void TestProcessGroupHelpers_CreatePipelineGroups()
        {
            // Act
            var groups = ProcessGroupHelpers.CreatePipelineGroups(_manager, 2);

            // Assert
            Assert.AreEqual(2, groups.Count);
            foreach (var kvp in groups)
            {
                Assert.IsNotNull(kvp.Value);
            }
        }
    }
}
```

### 6. Performance and Fault Tolerance Tests
Tests for performance optimizations and error handling.

```csharp
namespace MLFramework.Tests.Communication
{
    [TestFixture]
    public class PerformanceAndFaultToleranceTests
    {
        [Test]
        public void TestAlgorithmSelector_SelectsRingForLargeMessages()
        {
            // Arrange
            var selector = new AlgorithmSelector(8, new CommunicationConfig());

            // Act
            var algorithm = selector.SelectAllReduceAlgorithm(16 * 1024 * 1024); // 16 MB

            // Assert
            Assert.AreEqual(CommunicationAlgorithm.Ring, algorithm);
        }

        [Test]
        public void TestAlgorithmSelector_SelectsRecursiveDoublingForSmallMessages()
        {
            // Arrange
            var selector = new AlgorithmSelector(4, new CommunicationConfig());

            // Act
            var algorithm = selector.SelectAllReduceAlgorithm(2048); // 2 KB

            // Assert
            Assert.AreEqual(CommunicationAlgorithm.RecursiveDoubling, algorithm);
        }

        [Test]
        public void TestCommunicationProfiler_ProfilesOperations()
        {
            // Arrange
            var profiler = new CommunicationProfiler();
            var backend = new MockCommunicationBackend(0, 4);
            var tensor = CreateTestTensor(10);

            // Act
            profiler.Profile("AllReduce", 40, () => backend.AllReduce(tensor, ReduceOp.Sum));

            // Assert
            Assert.AreEqual(1, profiler.Profiles.Count);
            Assert.AreEqual("AllReduce", profiler.Profiles[0].Operation);
        }

        [Test]
        public void TestFaultTolerantCommunication_RetriesOnFailure()
        {
            // Arrange
            var backend = new MockCommunicationBackend(0, 4);
            backend.SetFailure(true, 2); // Fail twice, then succeed
            var ftBackend = new FaultTolerantCommunication(backend, new CommunicationConfig { TimeoutMs = 5000 });
            var tensor = CreateTestTensor(10);

            // Act & Assert - should not throw after retries
            Assert.DoesNotThrow(() => ftBackend.AllReduce(tensor, ReduceOp.Sum));
        }

        [Test]
        public void TestTimeoutManager_TimeoutsOperation()
        {
            // Arrange
            var manager = new TimeoutManager(100); // 100ms timeout
            var backend = new MockCommunicationBackend(0, 4);
            backend.SetDelay(500); // 500ms delay

            // Act
            var token = manager.StartTimeout(1, 100);
            var task = Task.Run(() => backend.AllReduce(CreateTestTensor(10), ReduceOp.Sum));

            // Assert - operation should timeout
            Assert.Throws<AggregateException>(() => task.Wait());
        }

        [Test]
        public void TestHealthMonitor_TracksRankStatus()
        {
            // Arrange
            var backend = new MockCommunicationBackend(0, 4);
            var monitor = new HealthMonitor(backend, TimeSpan.FromSeconds(1));

            // Act
            monitor.UpdateHeartbeat(0);
            var status = monitor.GetRankHealthStatus(0);

            // Assert
            Assert.AreEqual(RankHealthStatus.Healthy, status);
        }

        private Tensor<float> CreateTestTensor(int size)
        {
            var data = Enumerable.Range(0, size).Select(i => (float)i).ToArray();
            return new Tensor<float>(data);
        }
    }
}
```

## Implementation Notes

1. **File Structure:**
   - `tests/MLFramework.Tests/Communication/MockCommunicationBackend.cs`
   - `tests/MLFramework.Tests/Communication/CommunicationInterfaceTests.cs`
   - `tests/MLFramework.Tests/Communication/CollectiveOperationTests.cs`
   - `tests/MLFramework.Tests/Communication/PointToPointTests.cs`
   - `tests/MLFramework.Tests/Communication/ProcessGroupTests.cs`
   - `tests/MLFramework.Tests/Communication/PerformanceAndFaultToleranceTests.cs`

2. **Testing Strategy:**
   - Use mock backend to simulate communication without real hardware
   - Test all operations with various parameters
   - Test error conditions and edge cases
   - Test async operations
   - Test performance optimizations
   - Test fault tolerance mechanisms

3. **Test Coverage:**
   - All public methods tested
   - All exception paths tested
   - Edge cases covered
   - Integration between components tested

4. **Test Framework:**
   - Use NUnit or xUnit
   - Use Assert for assertions
   - Use SetUp/TearDown for test lifecycle
   - Use TestCase for parameterized tests

## Testing Requirements
- All tests pass with mock backend
- Code coverage > 80%
- No test relies on external hardware
- All tests run in < 5 seconds total

## Success Criteria
- Mock backend correctly simulates all operations
- All interface tests pass
- All collective operation tests pass
- All point-to-point tests pass
- All process group tests pass
- All performance and fault tolerance tests pass
- Test coverage meets requirements
