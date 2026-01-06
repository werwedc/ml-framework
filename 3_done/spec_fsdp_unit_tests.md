# Spec: FSDP Unit Tests

## Overview
Implement comprehensive unit tests for all FSDP components to ensure correctness and reliability.

## Requirements

### 1. FSDPConfig Tests
Test configuration validation and defaults:

```csharp
[TestClass]
public class FSDPConfigTests
{
    [TestMethod]
    public void TestDefaultConfig()
    {
        var config = new FSDPConfig();

        Assert.AreEqual(ShardingStrategy.Full, config.ShardingStrategy);
        Assert.IsTrue(config.MixedPrecision);
        Assert.IsFalse(config.OffloadToCPU);
        Assert.IsFalse(config.ActivationCheckpointing);
        Assert.AreEqual(25, config.BucketSizeMB);
        Assert.AreEqual(2, config.NumCommunicationWorkers);
    }

    [TestMethod]
    public void TestValidConfig()
    {
        var config = new FSDPConfig
        {
            ShardingStrategy = ShardingStrategy.LayerWise,
            BucketSizeMB = 50,
            NumCommunicationWorkers = 4
        };

        config.Validate(); // Should not throw
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void TestInvalidBucketSizeTooSmall()
    {
        var config = new FSDPConfig { BucketSizeMB = 0 };
        config.Validate();
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void TestInvalidBucketSizeTooLarge()
    {
        var config = new FSDPConfig { BucketSizeMB = 1001 };
        config.Validate();
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void TestInvalidCommunicationWorkers()
    {
        var config = new FSDPConfig { NumCommunicationWorkers = 0 };
        config.Validate();
    }
}
```

### 2. FSDPShardingUnit Tests
Test sharding unit operations:

```csharp
[TestClass]
public class FSDPShardingUnitTests
{
    private Mock<IProcessGroup> _mockProcessGroup;

    [TestInitialize]
    public void Setup()
    {
        _mockProcessGroup = new Mock<IProcessGroup>();
        _mockProcessGroup.Setup(p => p.Rank).Returns(0);
        _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
    }

    [TestMethod]
    public void TestShardingUnitCreation()
    {
        var param = Tensor.Zeros(new[] { 1000L }, TensorDataType.Float32);
        for (int i = 0; i < param.Size; i++)
        {
            param.Data[i] = i;
        }

        var unit = new FSDPShardingUnit("test_param", param, _mockProcessGroup.Object);

        Assert.AreEqual("test_param", unit.ParameterName);
        Assert.AreEqual(250, unit.ShardedParameter!.Size); // 1000 / 4
        Assert.AreEqual(0, unit.State.OwnerRank);
        Assert.AreEqual(4, unit.State.NumShards);
        Assert.IsFalse(unit.State.IsGathered);
        Assert.IsFalse(unit.State.IsOffloaded);
    }

    [TestMethod]
    public void TestShardingWithUnevenSize()
    {
        var param = Tensor.Zeros(new[] { 1003L }, TensorDataType.Float32);
        for (int i = 0; i < param.Size; i++)
        {
            param.Data[i] = i;
        }

        var unit = new FSDPShardingUnit("test_param", param, _mockProcessGroup.Object);

        // Rank 0 should get 251 elements (1003 / 4 + 1)
        Assert.AreEqual(251, unit.ShardedParameter!.Size);

        // Verify the shard contains the correct elements
        for (int i = 0; i < 251; i++)
        {
            Assert.AreEqual(i, unit.ShardedParameter.Data[i]);
        }
    }

    [TestMethod]
    public void TestReleaseGatheredParameters()
    {
        var param = Tensor.Zeros(new[] { 100L }, TensorDataType.Float32);
        var unit = new FSDPShardingUnit("test_param", param, _mockProcessGroup.Object);

        // Create a gathered parameter
        unit.GatheredParameter = Tensor.Zeros(new[] { 100L }, TensorDataType.Float32);
        Assert.IsNotNull(unit.GatheredParameter);

        // Release it
        unit.ReleaseGatheredParameters();
        Assert.IsNull(unit.GatheredParameter);
        Assert.IsFalse(unit.State.IsGathered);
    }

    [TestMethod]
    public void TestDisposal()
    {
        var param = Tensor.Zeros(new[] { 100L }, TensorDataType.Float32);
        var unit = new FSDPShardingUnit("test_param", param, _mockProcessGroup.Object);

        unit.Dispose();

        Assert.IsNull(unit.ShardedParameter);
        Assert.IsNull(unit.GatheredParameter);
        Assert.IsNull(unit.LocalGradient);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void TestEmptyParameterName()
    {
        var param = Tensor.Zeros(new[] { 100L }, TensorDataType.Float32);
        var unit = new FSDPShardingUnit("", param, _mockProcessGroup.Object);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void TestNullParameter()
    {
        var unit = new FSDPShardingUnit("test", null, _mockProcessGroup.Object);
    }
}
```

### 3. AllGatherOperation Tests
Test All-Gather communication:

```csharp
[TestClass]
public class AllGatherOperationTests
{
    private Mock<IProcessGroup> _mockProcessGroup;

    [TestInitialize]
    public void Setup()
    {
        _mockProcessGroup = new Mock<IProcessGroup>();
    }

    [TestMethod]
    public void TestSingleDeviceAllGather()
    {
        _mockProcessGroup.Setup(p => p.WorldSize).Returns(1);
        _mockProcessGroup.Setup(p => p.Rank).Returns(0);

        var shard = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f });
        var op = new AllGatherOperation(_mockProcessGroup.Object, new[] { 3L }, TensorDataType.Float32, 0);

        var result = op.AllGather(shard);

        Assert.AreEqual(3, result.Size);
        CollectionAssert.AreEqual(new[] { 1.0f, 2.0f, 3.0f }, result.Data);
    }

    [TestMethod]
    public void TestAllGatherMultipleDevices()
    {
        _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
        _mockProcessGroup.Setup(p => p.Rank).Returns(0);

        var shard = Tensor.FromArray(new[] { 1.0f, 2.0f });
        var op = new AllGatherOperation(_mockProcessGroup.Object, new[] { 8L }, TensorDataType.Float32, 0);

        // Note: This test will need to mock the Send/Recv operations
        // For now, we just test the object creation and basic validation
        Assert.IsNotNull(op);
        Assert.AreEqual(8, op.GetGatheredBuffer().Size);
    }

    [TestMethod]
    public void TestAllGatherUnevenShards()
    {
        _mockProcessGroup.Setup(p => p.WorldSize).Returns(3);
        _mockProcessGroup.Setup(p => p.Rank).Returns(0);

        var shard = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f });
        var op = new AllGatherOperation(_mockProcessGroup.Object, new[] { 10L }, TensorDataType.Float32, 0);

        // 10 elements / 3 devices = 4, 3, 3 elements per device
        Assert.IsNotNull(op);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void TestNullTensor()
    {
        _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
        _mockProcessGroup.Setup(p => p.Rank).Returns(0);

        var op = new AllGatherOperation(_mockProcessGroup.Object, new[] { 8L }, TensorDataType.Float32, 0);
        op.AllGather(null);
    }
}
```

### 4. ReduceScatterOperation Tests
Test Reduce-Scatter communication:

```csharp
[TestClass]
public class ReduceScatterOperationTests
{
    private Mock<IProcessGroup> _mockProcessGroup;

    [TestInitialize]
    public void Setup()
    {
        _mockProcessGroup = new Mock<IProcessGroup>();
    }

    [TestMethod]
    public void TestSingleDeviceReduceScatter()
    {
        _mockProcessGroup.Setup(p => p.WorldSize).Returns(1);
        _mockProcessGroup.Setup(p => p.Rank).Returns(0);

        var fullTensor = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f });
        var op = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 3L }, TensorDataType.Float32, 0);

        var result = op.ReduceScatter(fullTensor);

        Assert.AreEqual(3, result.Size);
        CollectionAssert.AreEqual(new[] { 1.0f, 2.0f, 3.0f }, result.Data);
    }

    [TestMethod]
    public void TestReduceScatterSum()
    {
        _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
        _mockProcessGroup.Setup(p => p.Rank).Returns(0);

        var fullTensor = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f });
        var op = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 4L }, TensorDataType.Float32, 0);

        // Test basic validation
        Assert.IsNotNull(op);
        Assert.AreEqual(ReduceOp.Sum, op.GetReduceOp());
    }

    [TestMethod]
    public void TestReduceScatterAvg()
    {
        _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
        _mockProcessGroup.Setup(p => p.Rank).Returns(0);

        var fullTensor = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f });
        var op = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 4L }, TensorDataType.Float32, 0, ReduceOp.Avg);

        Assert.AreEqual(ReduceOp.Avg, op.GetReduceOp());
    }

    [TestMethod]
    public void TestReduceScatterMax()
    {
        _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
        _mockProcessGroup.Setup(p => p.Rank).Returns(0);

        var fullTensor = Tensor.FromArray(new[] { 1.0f, 2.0f, 3.0f, 4.0f });
        var op = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 4L }, TensorDataType.Float32, 0, ReduceOp.Max);

        Assert.AreEqual(ReduceOp.Max, op.GetReduceOp());
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void TestNullTensor()
    {
        _mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
        _mockProcessGroup.Setup(p => p.Rank).Returns(0);

        var op = new ReduceScatterOperation(_mockProcessGroup.Object, new[] { 4L }, TensorDataType.Float32, 0);
        op.ReduceScatter(null);
    }
}
```

### 5. ShardingStrategy Tests
Test sharding strategy implementations:

```csharp
[TestClass]
public class ShardingStrategyTests
{
    [TestMethod]
    public void TestFullShardingStrategy()
    {
        var strategy = new FullShardingStrategy();

        var parameters = new List<ParameterInfo>
        {
            new ParameterInfo
            {
                Name = "param1",
                Shape = new[] { 1000L },
                SizeBytes = 4000,
                LayerName = "layer1",
                AlwaysGather = false
            },
            new ParameterInfo
            {
                Name = "param2",
                Shape = new[] { 500L },
                SizeBytes = 2000,
                LayerName = "layer1",
                AlwaysGather = false
            }
        };

        var plan = strategy.CalculateShardingPlan(parameters, 4);

        Assert.AreEqual(4, plan.TotalShards);
        Assert.IsTrue(plan.Assignments.Count > 0);
        Assert.AreEqual(0, plan.AlwaysGathered.Count);
    }

    [TestMethod]
    public void TestLayerWiseShardingStrategy()
    {
        var strategy = new LayerWiseShardingStrategy();

        var parameters = new List<ParameterInfo>
        {
            new ParameterInfo
            {
                Name = "layer1.weight",
                Shape = new[] { 1000L },
                SizeBytes = 4000,
                LayerName = "layer1",
                AlwaysGather = false
            },
            new ParameterInfo
            {
                Name = "layer2.weight",
                Shape = new[] { 500L },
                SizeBytes = 2000,
                LayerName = "layer2",
                AlwaysGather = false
            }
        };

        var plan = strategy.CalculateShardingPlan(parameters, 4);

        Assert.AreEqual(4, plan.TotalShards);
        Assert.IsTrue(plan.Assignments.Count > 0);
    }

    [TestMethod]
    public void TestHybridShardingStrategy()
    {
        var fullShardedLayers = new List<string> { "transformer" };
        var layerWiseShardedLayers = new List<string> { "classifier" };

        var strategy = new HybridShardingStrategy(fullShardedLayers, layerWiseShardedLayers);

        var parameters = new List<ParameterInfo>
        {
            new ParameterInfo
            {
                Name = "transformer.weight",
                Shape = new[] { 1000L },
                SizeBytes = 4000,
                LayerName = "transformer",
                AlwaysGather = false
            },
            new ParameterInfo
            {
                Name = "classifier.weight",
                Shape = new[] { 500L },
                SizeBytes = 2000,
                LayerName = "classifier",
                AlwaysGather = false
            }
        };

        var plan = strategy.CalculateShardingPlan(parameters, 4);

        Assert.AreEqual(4, plan.TotalShards);
        Assert.IsTrue(plan.Assignments.Count > 0);
    }

    [TestMethod]
    public void TestAlwaysGatheredParameters()
    {
        var strategy = new FullShardingStrategy();

        var parameters = new List<ParameterInfo>
        {
            new ParameterInfo
            {
                Name = "embedding.weight",
                Shape = new[] { 10000L },
                SizeBytes = 40000,
                LayerName = "embedding",
                AlwaysGather = true
            }
        };

        var plan = strategy.CalculateShardingPlan(parameters, 4);

        Assert.AreEqual(1, plan.AlwaysGathered.Count);
        Assert.IsTrue(plan.AlwaysGathered.Contains("embedding.weight"));
        Assert.IsFalse(plan.Assignments.ContainsKey("embedding.weight"));
    }
}
```

### 6. FSDPOptimizerState Tests
Test optimizer state management:

```csharp
[TestClass]
public class FSDPOptimizerStateTests
{
    [TestMethod]
    public void TestAdamOptimizerStateCreation()
    {
        var param = Tensor.Zeros(new[] { 100L }, TensorDataType.Float32);
        var state = new AdamOptimizerState(param, 0, 4);

        Assert.AreEqual(OptimizerStateType.Adam, state.StateType);
        Assert.AreEqual(0, state.ShardIndex);
        Assert.AreEqual(4, state.NumShards);
        Assert.AreEqual(0, state.StepCount);
        Assert.IsNotNull(state.MomentumBuffer);
        Assert.IsNotNull(state.VarianceBuffer);
        Assert.AreEqual(100, state.MomentumBuffer.Size);
        Assert.AreEqual(100, state.VarianceBuffer.Size);
    }

    [TestMethod]
    public void TestSGDOptimizerStateCreation()
    {
        var state = new SGDOptimizerState(0, 4);

        Assert.AreEqual(OptimizerStateType.SGD, state.StateType);
        Assert.AreEqual(0, state.ShardIndex);
        Assert.AreEqual(4, state.NumShards);
    }

    [TestMethod]
    public void TestAdamOptimizerStateCloning()
    {
        var param = Tensor.Zeros(new[] { 100L }, TensorDataType.Float32);
        var state = new AdamOptimizerState(param, 0, 4);
        state.StepCount = 10;

        var cloned = state.Clone() as AdamOptimizerState;

        Assert.IsNotNull(cloned);
        Assert.AreEqual(state.ShardIndex, cloned.ShardIndex);
        Assert.AreEqual(state.NumShards, cloned.NumShards);
        Assert.AreEqual(state.StepCount, cloned.StepCount);
        Assert.AreEqual(state.MomentumBuffer.Size, cloned.MomentumBuffer.Size);
        Assert.AreEqual(state.VarianceBuffer.Size, cloned.VarianceBuffer.Size);
    }

    [TestMethod]
    public void TestOptimizerStateDisposal()
    {
        var param = Tensor.Zeros(new[] { 100L }, TensorDataType.Float32);
        var state = new AdamOptimizerState(param, 0, 4);

        state.Dispose();

        Assert.IsNull(state.MomentumBuffer);
        Assert.IsNull(state.VarianceBuffer);
    }
}
```

### 7. Integration Tests
Test end-to-end FSDP functionality:

```csharp
[TestClass]
public class FSDPIntegrationTests
{
    [TestMethod]
    public void TestFSDPWrapperCreation()
    {
        var mockModel = new Mock<IModel>();
        var mockProcessGroup = new Mock<IProcessGroup>();

        mockProcessGroup.Setup(p => p.WorldSize).Returns(4);
        mockProcessGroup.Setup(p => p.Rank).Returns(0);
        mockProcessGroup.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());

        var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

        var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);

        Assert.IsNotNull(fsdp);
    }

    [TestMethod]
    public void TestFSDPWithSingleDevice()
    {
        var mockModel = new Mock<IModel>();
        var mockProcessGroup = new Mock<IProcessGroup>();

        mockProcessGroup.Setup(p => p.WorldSize).Returns(1);
        mockProcessGroup.Setup(p => p.Rank).Returns(0);
        mockProcessGroup.Setup(p => p.Backend).Returns(Mock.Of<ICommunicationBackend>());

        var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

        // Should not throw for single device
        var fsdp = new FSDP(mockModel.Object, config, mockProcessGroup.Object);
        Assert.IsNotNull(fsdp);
    }

    [TestMethod]
    [ExpectedException(typeof(InvalidOperationException))]
    public void TestFSDPWithoutProcessGroup()
    {
        var mockModel = new Mock<IModel>();
        ProcessGroup.Destroy(); // Ensure no default process group

        var config = new FSDPConfig { ShardingStrategy = ShardingStrategy.Full };

        var fsdp = new FSDP(mockModel.Object, config);
    }
}
```

## Directory Structure
- **File**: `tests/MLFramework.Distributed.FSDP.Tests/FSDPConfigTests.cs`
- **File**: `tests/MLFramework.Distributed.FSDP.Tests/FSDPShardingUnitTests.cs`
- **File**: `tests/MLFramework.Distributed.FSDP.Tests/AllGatherOperationTests.cs`
- **File**: `tests/MLFramework.Distributed.FSDP.Tests/ReduceScatterOperationTests.cs`
- **File**: `tests/MLFramework.Distributed.FSDP.Tests/ShardingStrategyTests.cs`
- **File**: `tests/MLFramework.Distributed.FSDP.Tests/FSDPOptimizerStateTests.cs`
- **File**: `tests/MLFramework.Distributed.FSDP.Tests/FSDPIntegrationTests.cs`
- **Namespace**: `MLFramework.Distributed.FSDP.Tests`

## Dependencies
- `MLFramework.Distributed.FSDP`
- `MLFramework.Distributed.FSDP.*` (all components)
- `Moq` (for mocking)
- `MSTest` (or NUnit, xUnit)

## Implementation Notes
1. Use Moq framework for mocking IProcessGroup and other interfaces
2. Test edge cases (single device, empty parameters, null values)
3. Test error handling (invalid configurations, failed operations)
4. Include integration tests for end-to-end functionality
5. Test async operations where applicable

## Estimated Time
45 minutes
