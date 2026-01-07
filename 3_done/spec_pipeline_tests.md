# Spec: Pipeline Parallelism Tests

## Overview
Comprehensive test suite for pipeline parallelism, including unit tests, integration tests, and end-to-end tests that verify correctness and performance.

## Test Structure

### Test Categories

1. **Unit Tests** - Test individual components in isolation
2. **Integration Tests** - Test component interactions
3. **End-to-End Tests** - Test full pipeline training scenarios
4. **Performance Tests** - Measure throughput and efficiency
5. **Regression Tests** - Ensure correctness against baseline

## Test Files

### 1. PipelineStageTests.cs
```csharp
using MLFramework.Pipeline;
using Xunit;

public class PipelineStageTests
{
    [Fact]
    public void Constructor_WithValidInputs_CreatesStage()
    {
        // Test constructor creates valid PipelineStage
    }

    [Fact]
    public void Constructor_WithInvalidRank_Throws()
    {
        // Test throws on invalid rank
    }

    [Fact]
    public void Forward_PassesThroughModule()
    {
        // Test forward pass works
    }

    [Fact]
    public void IsFirstStage_ReturnsCorrectValue()
    {
        // Test property correctness
    }

    // ... more tests
}
```

### 2. LayerPartitionerTests.cs
```csharp
public class LayerPartitionerTests
{
    [Fact]
    public void UniformPartition_WithEvenLayers_CreatesEqualStages()
    {
        // Test uniform partitioning
    }

    [Fact]
    public void UniformPartition_WithUnevenLayers_DistributesCorrectly()
    {
        // Test handles remainder
    }

    [Fact]
    public void ManualPartition_WithValidInput_CreatesSpecifiedStages()
    {
        // Test manual partitioning
    }

    [Fact]
    public void AutomaticPartition_BalancesMemory()
    {
        // Test auto partitioning
    }

    // ... more tests
}
```

### 3. PipelineCommunicatorTests.cs
```csharp
public class PipelineCommunicatorTests
{
    [Fact]
    public async Task SendAndReceive_BetweenTwoRanks_WorksCorrectly()
    {
        // Test basic communication
    }

    [Fact]
    public async Task MultipleSequentialSends_WorksCorrectly()
    {
        // Test sequential operations
    }

    [Fact]
    public async Task Barrier_SynchronizesProcesses()
    {
        // Test barrier synchronization
    }

    // ... more tests
}
```

### 4. MicroBatchManagerTests.cs
```csharp
public class MicroBatchManagerTests
{
    [Fact]
    public void SplitBatch_WithEvenDivision_CreatesCorrectMicroBatches()
    {
        // Test batch splitting
    }

    [Fact]
    public void AccumulateGradients_AveragesCorrectly()
    {
        // Test gradient accumulation
    }

    [Fact]
    public void ResetGradients_ClearsAccumulatedGradients()
    {
        // Test reset functionality
    }

    // ... more tests
}
```

### 5. GPipeSchedulerTests.cs
```csharp
public class GPipeSchedulerTests
{
    [Fact]
    public void Forward_WithTwoStages_ProducesCorrectOutput()
    {
        // Test forward pass
    }

    [Fact]
    public void Backward_WithTwoStages_ProducesCorrectGradients()
    {
        // Test backward pass
    }

    [Fact]
    public void TrainIteration_WithSimpleModel_CompletesSuccessfully()
    {
        // Test full iteration
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    public void Forward_WithMultipleStages_WorksCorrectly(int numStages)
    {
        // Test with varying stage counts
    }

    // ... more tests
}
```

### 6. ActivationCheckpointManagerTests.cs
```csharp
public class ActivationCheckpointManagerTests
{
    [Fact]
    public void StoreAllStrategy_CheckpointsAllActivations()
    {
        // Test StoreAll strategy
    }

    [Fact]
    public void RecomputeAllStrategy_CheckpointsNone()
    {
        // Test RecomputeAll strategy
    }

    [Fact]
    public void GetOrCompute_WithStoredActivation_ReturnsStored()
    {
        // Test retrieval
    }

    [Fact]
    public void RecomputeActivation_ProducesCorrectResult()
    {
        // Test recomputation
    }

    // ... more tests
}
```

### 7. PipelineOptimizerTests.cs
```csharp
public class PipelineOptimizerTests
{
    [Fact]
    public void Step_WithSynchronization_UpdatesAllStages()
    {
        // Test optimizer step
    }

    [Fact]
    public void SynchronizeGradients_AveragesCorrectly()
    {
        // Test gradient sync
    }

    [Fact]
    public void BroadcastParameters_SyncsAllStages()
    {
        // Test parameter broadcast
    }

    [Fact]
    public void LearningRate_GetterSetter_WorksCorrectly()
    {
        // Test LR management
    }

    // ... more tests
}
```

### 8. AsyncPipelineExecutorTests.cs
```csharp
public class AsyncPipelineExecutorTests
{
    [Fact]
    public async Task ForwardAsync_ExecutesSuccessfully()
    {
        // Test async forward
    }

    [Fact]
    public async Task OverlappedComputeAndComm_WorksCorrectly()
    {
        // Test overlapping
    }

    [Fact]
    public async Task SyncAll_WaitsForAllStreams()
    {
        // Test synchronization
    }

    // ... more tests
}
```

### 9. PipelineValidatorTests.cs
```csharp
public class PipelineValidatorTests
{
    [Fact]
    public void ValidateConfiguration_WithValidSetup_ReturnsTrue()
    {
        // Test config validation
    }

    [Fact]
    public void ValidateParameterConsistency_WithMismatch_ReturnsFalse()
    {
        // Test parameter check
    }

    [Fact]
    public void ValidateNumericalStability_DetectsNaN_ReportsError()
    {
        // Test NaN detection
    }

    // ... more tests
}
```

## Integration Test Suite

### End-to-End Training Tests

```csharp
public class PipelineEndToEndTests
{
    [Fact]
    public async Task TrainSimpleMLP_WithPipeline_Converges()
    {
        // Create simple MLP model
        // Partition into 4 stages
        // Train for 10 iterations
        // Verify loss decreases
    }

    [Fact]
    public async Task TrainCNN_WithPipeline_ProducesCorrectGradients()
    {
        // Create CNN model
        // Train one iteration
        // Compare gradients with single-device baseline
        // Assert gradients match within tolerance
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    public async Task TrainModel_WithDifferentNumStages_WorksCorrectly(int numStages)
    {
        // Test with varying stage counts
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    public async Task TrainModel_WithDifferentMicroBatches_WorksCorrectly(int microBatches)
    {
        // Test with varying micro-batch counts
    }
}
```

### Correctness Tests

```csharp
public class PipelineCorrectnessTests
{
    [Fact]
    public async Task PipelineVsSingleDevice_ProduceSameGradients()
    {
        // Train one step with pipeline
        // Train one step with single device
        // Compare gradients
        // Assert difference < 1e-4
    }

    [Fact]
    public async Task PipelineVsSingleDevice_ProduceSameLoss()
    {
        // Compare loss values
        // Assert difference < 1e-6
    }

    [Fact]
    public async Task MultipleIterations_ProduceConsistentResults()
    {
        // Run multiple iterations
        // Verify reproducibility
    }
}
```

## Performance Tests

```csharp
public class PipelinePerformanceTests
{
    [Fact]
    public async Task MeasureThroughput_ReportsTokensPerSecond()
    {
        // Measure tokens/sec
        // Log to console
        // No assertion (just reporting)
    }

    [Fact]
    public async Task MeasureMemoryUsage_ReportsPerStageMemory()
    {
        // Measure memory per stage
        // Verify < threshold
    }

    [Fact]
    public async Task MeasureUtilization_ReportsDeviceUtilization()
    {
        // Measure bubble time
        // Assert utilization > threshold (e.g., 70%)
    }

    [Fact]
    public async Task CompareSpeedup_VaryingStages_MeasuresScaling()
    {
        // Measure speedup with 2, 4, 8 stages
        // Verify near-linear scaling
    }
}
```

## Test Utilities

### TestHelper Class

```csharp
public static class TestHelper
{
    /// <summary>
    /// Create a simple MLP model for testing
    /// </summary>
    public static Module CreateSimpleMLP(int inputSize, int hiddenSize, int outputSize);

    /// <summary>
    /// Create a simple CNN model for testing
    /// </summary>
    public static Module CreateSimpleCNN(int inputChannels, int numClasses);

    /// <summary>
    /// Create dummy input tensor
    /// </summary>
    public static Tensor CreateDummyInput(int batchSize, int inputSize, Device device);

    /// <summary>
    /// Compare two tensors (assert they match within tolerance)
    /// </summary>
    public static void AssertTensorClose(Tensor a, Tensor b, float tolerance = 1e-4);

    /// <summary>
    /// Run both pipeline and single-device and compare results
    /// </summary>
    public static (Tensor pipelineLoss, Tensor singleLoss) CompareWithBaseline(
        Module model,
        Tensor input,
        int numStages,
        int microBatches);

    /// <summary>
    /// Measure execution time of an action
    /// </summary>
    public static long MeasureExecutionTime(Action action);
}
```

## Test Configuration

### Test Models

Use simple but realistic test models:
- **Small MLP**: 2-3 hidden layers, 64-128 units
- **Small CNN**: 2 conv layers, 1-2 fc layers
- **Transformer Block**: 1-2 attention layers

### Test Data

Use random tensors with reasonable dimensions:
- Input batches: 32-128 samples
- Input sizes: 784 (MNIST), 3072 (CIFAR), etc.
- Random initialization with fixed seed for reproducibility

## Running Tests

```bash
# Run all pipeline tests
dotnet test --filter "FullyQualifiedName~MLFramework.Pipeline"

# Run specific test
dotnet test --filter "FullyQualifiedName~GPipeSchedulerTests.Forward_WithTwoStages"

# Run performance tests (requires --no-build and specific config)
dotnet test --filter "FullyQualifiedName~PipelinePerformanceTests" --configuration Release
```

## Files to Create

- `tests/Pipeline/PipelineStageTests.cs`
- `tests/Pipeline/LayerPartitionerTests.cs`
- `tests/Pipeline/PipelineCommunicatorTests.cs`
- `tests/Pipeline/MicroBatchManagerTests.cs`
- `tests/Pipeline/GPipeSchedulerTests.cs`
- `tests/Pipeline/ActivationCheckpointManagerTests.cs`
- `tests/Pipeline/PipelineOptimizerTests.cs`
- `tests/Pipeline/AsyncPipelineExecutorTests.cs`
- `tests/Pipeline/PipelineValidatorTests.cs`
- `tests/Pipeline/PipelineEndToEndTests.cs`
- `tests/Pipeline/PipelineCorrectnessTests.cs`
- `tests/Pipeline/PipelinePerformanceTests.cs`
- `tests/Pipeline/TestHelper.cs`

## Dependencies

- xUnit for testing framework
- All previous pipeline specs
- Existing ML framework components

## Time Estimate

60-90 minutes for comprehensive test suite

## Notes

- Prioritize correctness tests over performance tests
- Use fixed random seeds for reproducibility
- Make tests fast (avoid large models/long training)
- Parallelize independent tests where possible
- Document any known test flakiness
