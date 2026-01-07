# Spec: DDP Test Suite

## Overview
Implement comprehensive tests for the Distributed Data Parallelism feature, covering all components from communication backends to the high-level DDP module.

## Requirements
- Unit tests for each major component
- Integration tests for end-to-end functionality
- Performance benchmarks
- Mock implementations for testing without actual distributed hardware
- Test coverage for edge cases and error scenarios

## Test Structure

### 1. Test Organization

```
tests/
├── Distributed/
│   ├── CommunicationBackendTests.cs
│   ├── ProcessGroupTests.cs
│   ├── RingAllReduceTests.cs
│   ├── DistributedSamplerTests.cs
│   ├── GradientBucketingTests.cs
│   ├── DDPModuleTests.cs
│   ├── NCCLBackendTests.cs
│   ├── GlooBackendTests.cs
│   ├── ProcessLauncherTests.cs
│   └── Integration/
│       ├── EndToEndTests.cs
│       └── PerformanceTests.cs
```

### 2. Test Utilities

#### MockProcessGroup.cs
```csharp
/// <summary>
/// Mock process group for testing without actual distributed hardware.
/// </summary>
public class MockProcessGroup : IProcessGroup
{
    private readonly int _rank;
    private readonly int _worldSize;
    private readonly Dictionary<int, List<Tensor>> _mockState;

    public MockProcessGroup(int rank, int worldSize)
    {
        _rank = rank;
        _worldSize = worldSize;
        _mockState = new Dictionary<int, List<Tensor>>();
        for (int i = 0; i < worldSize; i++)
        {
            _mockState[i] = new List<Tensor>();
        }
    }

    public int Rank => _rank;
    public int WorldSize => _worldSize;
    public ICommunicationBackend Backend => new MockBackend();

    public void AllReduce(Tensor tensor, ReduceOp op)
    {
        // Store tensor for this rank
        _mockState[_rank].Add(tensor.Clone());

        // Simulate reduction by summing all tensors (simplified)
        var allTensors = new List<Tensor>();
        for (int i = 0; i < _worldSize; i++)
        {
            if (_mockState[i].Count > _rank)
            {
                allTensors.Add(_mockState[i][_rank]);
            }
        }

        if (allTensors.Count == _worldSize)
        {
            // All ranks have provided their tensor, perform reduction
            var sum = allTensors[0].Clone();
            for (int i = 1; i < allTensors.Count; i++)
            {
                sum.Add_(allTensors[i]);
            }

            if (op == ReduceOp.Avg)
            {
                sum.Div_(_worldSize);
            }

            // All ranks get the reduced result
            tensor.Copy_(sum);
        }
    }

    // Implement other methods similarly...

    public void Broadcast(Tensor tensor, int root)
    {
        // Simplified: copy tensor from root to all ranks
        // In tests, we can just use shared state
    }

    public void Barrier()
    {
        // No-op in mock
    }

    // Async versions just wrap sync versions
    public Task AllReduceAsync(Tensor tensor, ReduceOp op)
    {
        AllReduce(tensor, op);
        return Task.CompletedTask;
    }

    public Task BroadcastAsync(Tensor tensor, int root)
    {
        Broadcast(tensor, root);
        return Task.CompletedTask;
    }

    public Task BarrierAsync()
    {
        Barrier();
        return Task.CompletedTask;
    }

    public void Destroy() { }

    /// <summary>
    /// Helper to simulate all ranks providing tensors.
    /// </summary>
    public void SimulateAllReduce(List<Tensor> tensors, ReduceOp op)
    {
        var sum = tensors[0].Clone();
        for (int i = 1; i < tensors.Count; i++)
        {
            sum.Add_(tensors[i]);
        }

        if (op == ReduceOp.Avg)
        {
            sum.Div_(tensors.Count);
        }

        // Return reduced tensor to all ranks
        foreach (var tensor in tensors)
        {
            tensor.Copy_(sum);
        }
    }
}
```

#### TestDataset.cs
```csharp
/// <summary>
/// Simple dataset for testing.
/// </summary>
public class TestDataset : Dataset
{
    private readonly int _size;
    private readonly Func<int, Tensor> _dataGenerator;

    public TestDataset(int size, Func<int, Tensor> dataGenerator = null)
    {
        _size = size;
        _dataGenerator = dataGenerator ?? (i => Tensor.Random(10, 10));
    }

    public override int Count => _size;

    public override Tensor GetItem(int index)
    {
        return _dataGenerator(index);
    }
}
```

## Test Suites

### 1. CommunicationBackendTests.cs

```csharp
[TestClass]
public class CommunicationBackendTests
{
    [TestMethod]
    public void MockBackend_Availability_ReturnsTrue()
    {
        var backend = new MockBackend();
        Assert.IsTrue(backend.IsAvailable);
    }

    [TestMethod]
    public void ProcessGroup_Singleton_OnlyOneActiveGroup()
    {
        // Should not be able to create multiple active process groups
        var group1 = MockProcessGroup.Create(worldSize: 2, rank: 0);

        Assert.ThrowsException<InvalidOperationException>(() =>
        {
            var group2 = MockProcessGroup.Create(worldSize: 2, rank: 1);
        });

        group1.Destroy();
    }

    [TestMethod]
    public void ProcessGroup_RankAndWorldSize_AreCorrect()
    {
        var group = MockProcessGroup.Create(worldSize: 4, rank: 2);
        Assert.AreEqual(2, group.Rank);
        Assert.AreEqual(4, group.WorldSize);
        group.Destroy();
    }

    [TestMethod]
    public void ProcessGroup_Destroy_AllowsNewCreation()
    {
        var group1 = MockProcessGroup.Create(worldSize: 2, rank: 0);
        group1.Destroy();

        var group2 = MockProcessGroup.Create(worldSize: 2, rank: 0);
        Assert.IsNotNull(group2);
        group2.Destroy();
    }
}
```

### 2. RingAllReduceTests.cs

```csharp
[TestClass]
public class RingAllReduceTests
{
    private MockProcessGroup _processGroup;

    [TestInitialize]
    public void Setup()
    {
        _processGroup = MockProcessGroup.Create(worldSize: 4, rank: 0);
    }

    [TestCleanup]
    public void Cleanup()
    {
        _processGroup?.Destroy();
    }

    [TestMethod]
    public void RingAllReduce_Sum_CorrectlyAggregatesGradients()
    {
        var allReduce = new RingAllReduce(_processGroup);

        // Simulate 4 ranks with different gradients
        var gradients = new[]
        {
            Tensor.Ones(10),    // Rank 0
            Tensor.Ones(10).Mul(2),  // Rank 1
            Tensor.Ones(10).Mul(3),  // Rank 2
            Tensor.Ones(10).Mul(4)   // Rank 3
        };

        // Expected sum: 1 + 2 + 3 + 4 = 10
        var expected = Tensor.Ones(10).Mul(10);

        _processGroup.SimulateAllReduce(gradients, ReduceOp.Sum);

        for (int i = 0; i < gradients.Length; i++)
        {
            Assert.IsTrue(Tensor.AllClose(gradients[i], expected));
        }
    }

    [TestMethod]
    public void RingAllReduce_Avg_CorrectlyComputesAverage()
    {
        var allReduce = new RingAllReduce(_processGroup);
        var gradients = new[]
        {
            Tensor.Ones(10),
            Tensor.Ones(10).Mul(3),
            Tensor.Ones(10).Mul(5)
        };

        // Expected avg: (1 + 3 + 5) / 3 = 3
        var expected = Tensor.Ones(10).Mul(3);

        _processGroup.SimulateAllReduce(gradients, ReduceOp.Avg);

        foreach (var grad in gradients)
        {
            Assert.IsTrue(Tensor.AllClose(grad, expected));
        }
    }

    [TestMethod]
    public void RingAllReduce_Max_CorrectlyFindsMaximum()
    {
        var gradients = new[]
        {
            Tensor.Random(10).Mul(10),
            Tensor.Random(10).Mul(10),
            Tensor.Random(10).Mul(10)
        };

        var expected = Tensor.Maximum(Tensor.Maximum(gradients[0], gradients[1]), gradients[2]);

        _processGroup.SimulateAllReduce(gradients, ReduceOp.Max);

        foreach (var grad in gradients)
        {
            Assert.IsTrue(Tensor.AllClose(grad, expected));
        }
    }

    [TestMethod]
    public void RingAllReduce_Async_CompletesSuccessfully()
    {
        var allReduce = new RingAllReduce(_processGroup);
        var tensor = Tensor.Random(1000);

        var task = allReduce.AllReduceAsync(tensor, ReduceOp.Sum);
        Assert.IsTrue(task.IsCompleted);
    }

    [TestMethod]
    public void RingAllReduce_SingleDevice_SkipsCommunication()
    {
        var singleDeviceGroup = MockProcessGroup.Create(worldSize: 1, rank: 0);
        var allReduce = new RingAllReduce(singleDeviceGroup);
        var tensor = Tensor.Random(10);

        var original = tensor.Clone();
        allReduce.AllReduce(tensor, ReduceOp.Sum);

        // Tensor should be unchanged (single device)
        Assert.IsTrue(Tensor.AllClose(tensor, original));

        singleDeviceGroup.Destroy();
    }
}
```

### 3. DistributedSamplerTests.cs

```csharp
[TestClass]
public class DistributedSamplerTests
{
    [TestMethod]
    public void DistributedSampler_Partitions_CorrectlyAcrossRanks()
    {
        var dataset = new TestDataset(100);
        var worldSize = 4;

        var allIndices = new List<int>[worldSize];
        for (int rank = 0; rank < worldSize; rank++)
        {
            var sampler = new DistributedSampler(dataset, numReplicas: worldSize, rank: rank, shuffle: false);
            allIndices[rank] = sampler.GetIndices().ToList();
        }

        // Check that all indices are covered without overlap
        var allCombined = allIndices.SelectMany(i => i).OrderBy(i => i).ToList();
        var expected = Enumerable.Range(0, 100).ToList();

        CollectionAssert.AreEqual(expected, allCombined);

        // Check no overlaps
        var distinctCount = allCombined.Distinct().Count();
        Assert.AreEqual(100, distinctCount);
    }

    [TestMethod]
    public void DistributedSampler_SetEpoch_ChangesShuffleOrder()
    {
        var dataset = new TestDataset(100);
        var sampler1 = new DistributedSampler(dataset, numReplicas: 2, rank: 0, shuffle: true, seed: 42);
        sampler1.SetEpoch(0);
        var indices1 = sampler1.GetIndices();

        var sampler2 = new DistributedSampler(dataset, numReplicas: 2, rank: 0, shuffle: true, seed: 42);
        sampler2.SetEpoch(1);
        var indices2 = sampler2.GetIndices();

        // Should be different due to different epochs
        CollectionAssert.AreNotEqual(indices1, indices2);
    }

    [TestMethod]
    public void DistributedSampler_SameEpochSameSeed_SameOrder()
    {
        var dataset = new TestDataset(100);
        var sampler1 = new DistributedSampler(dataset, numReplicas: 2, rank: 0, shuffle: true, seed: 42);
        sampler1.SetEpoch(0);
        var indices1 = sampler1.GetIndices();

        var sampler2 = new DistributedSampler(dataset, numReplicas: 2, rank: 0, shuffle: true, seed: 42);
        sampler2.SetEpoch(0);
        var indices2 = sampler2.GetIndices();

        // Should be identical
        CollectionAssert.AreEqual(indices1, indices2);
    }
}
```

### 4. GradientBucketingTests.cs

```csharp
[TestClass]
public class GradientBucketingTests
{
    private MockProcessGroup _processGroup;

    [TestInitialize]
    public void Setup()
    {
        _processGroup = MockProcessGroup.Create(worldSize: 4, rank: 0);
    }

    [TestCleanup]
    public void Cleanup()
    {
        _processGroup?.Destroy();
    }

    [TestMethod]
    public void GradientBucketManager_CreatesBuckets_OfCorrectSize()
    {
        var parameters = new[]
        {
            Tensor.Random(10 * 1024 * 1024),  // ~40MB
            Tensor.Random(5 * 1024 * 1024),   // ~20MB
            Tensor.Random(15 * 1024 * 1024), // ~60MB
            Tensor.Random(3 * 1024 * 1024)    // ~12MB
        };

        var bucketManager = new GradientBucketManager(_processGroup, parameters, bucketSizeInBytes: 25 * 1024 * 1024);

        // Should create multiple buckets
        Assert.IsTrue(bucketManager.NumBuckets > 1);
    }

    [TestMethod]
    public void GradientBucketManager_Reduce_CorrectlyCopiesBack()
    {
        var gradients = new[] { Tensor.Random(100), Tensor.Random(100) };
        var bucketManager = new GradientBucketManager(_processGroup, gradients);

        var original1 = gradients[0].Clone();
        var original2 = gradients[1].Clone();

        // Simulate reduction (multiply by world size)
        _processGroup.SimulateAllReduce(gradients.ToList(), ReduceOp.Sum);
        bucketManager.CopyBackAll();

        // Check that gradients are reduced
        Assert.IsFalse(Tensor.AllClose(gradients[0], original1));
        Assert.IsFalse(Tensor.AllClose(gradients[1], original2));
    }
}
```

### 5. DDPModuleTests.cs

```csharp
[TestClass]
public class DDPModuleTests
{
    [TestMethod]
    public void DDP_Forward_PreservesModuleBehavior()
    {
        var processGroup = MockProcessGroup.Create(worldSize: 2, rank: 0);
        var model = new SimpleModel();
        var ddpModel = new DistributedDataParallel(model, processGroup);

        var input = Tensor.Random(10, 10);
        var output = ddpModel.Forward(input);

        Assert.IsNotNull(output);

        processGroup.Destroy();
    }

    [TestMethod]
    public void DDP_BroadcastParameters_SynchronizesWeights()
    {
        var processGroup1 = MockProcessGroup.Create(worldSize: 2, rank: 0);
        var processGroup2 = MockProcessGroup.Create(worldSize: 2, rank: 1);

        var model1 = new SimpleModel();
        var model2 = new SimpleModel();

        // Make models have different weights
        model1.Weight.Fill_(1.0);
        model2.Weight.Fill_(2.0);

        var ddp1 = new DistributedDataParallel(model1, processGroup1);
        var ddp2 = new DistributedDataParallel(model2, processGroup2);

        // Broadcast from rank 0
        ddp1.BroadcastParameters();
        ddp2.BroadcastParameters();

        // Both models should now have the same weights (from rank 0)
        Assert.IsTrue(Tensor.AllClose(model1.Weight, model2.Weight));

        processGroup1.Destroy();
        processGroup2.Destroy();
    }
}
```

### 6. EndToEndTests.cs

```csharp
[TestClass]
public class EndToEndTests
{
    [TestMethod]
    public void DistributedTraining_TrainsModel_WithSpeedup()
    {
        // This test requires actual distributed hardware or extensive mocking
        // For now, we can test the pipeline with mocked communication

        var processGroup = MockProcessGroup.Create(worldSize: 4, rank: 0);
        var model = new SimpleModel();
        var ddpModel = new DistributedDataParallel(model, processGroup);

        var dataset = new TestDataset(100, i => Tensor.Random(10, 10));
        var sampler = new DistributedSampler(dataset, numReplicas: 4, rank: 0, shuffle: true);
        var loader = new DataLoader(dataset, batchSize: 32, sampler: sampler);

        var optimizer = new SGD(model.GetParameters(), lr: 0.01);

        // Simulate training
        for (int epoch = 0; epoch < 2; epoch++)
        {
            sampler.SetEpoch(epoch);

            foreach (var batch in loader)
            {
                var output = ddpModel.Forward(batch);
                var loss = output.Mean();
                loss.Backward();
                optimizer.Step();
                optimizer.ZeroGrad();
            }
        }

        // If we reach here, training completed without errors
        Assert.IsTrue(true);

        processGroup.Destroy();
    }
}
```

## Test Coverage Goals

- **Communication Backend**: 95% coverage (interfaces, process group, mock)
- **Ring-AllReduce**: 90% coverage (correctness, edge cases, async)
- **Distributed Sampler**: 90% coverage (partitioning, shuffling, epochs)
- **Gradient Bucketing**: 85% coverage (bucket creation, reduction, copy back)
- **DDP Module**: 85% coverage (forward, backward, synchronization)
- **Backends**: 70% coverage (NCCL, Gloo - limited by hardware availability)
- **Process Launcher**: 80% coverage (environment setup, process management)

## Running Tests

### Single GPU / Mock Tests
```bash
dotnet test tests/Distributed.Tests/ --filter "FullyQualifiedName!~Integration"
```

### Integration Tests (Requires Multi-GPU)
```bash
# Set environment for distributed testing
export RANK=0
export WORLD_SIZE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

dotnet test tests/Distributed.Tests/ --filter "FullyQualifiedName~Integration"
```

### Performance Benchmarks
```bash
dotnet test tests/Distributed.Tests/ --filter "Performance"
```

## Success Criteria
- [ ] All unit tests pass with mocked components
- [ ] Integration tests pass on actual multi-GPU hardware (if available)
- [ ] Test coverage meets the goals above
- [ ] Performance benchmarks show expected speedup
- [ ] Tests catch common bugs and edge cases

## Dependencies
- All other DDP specs (tests verify their implementations)
- xUnit or NUnit testing framework
- Existing test utilities from the framework

## Notes

- Mock implementations are crucial for testing without distributed hardware
- Integration tests should be marked as [Ignore] or run conditionally
- Performance tests should not run by default in CI/CD
- Consider using test containers for multi-node testing
