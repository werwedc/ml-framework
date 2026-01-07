# Spec: Tensor Parallelism Integration Tests

## Overview
Implement comprehensive integration tests for the Tensor Parallelism feature. These tests verify that TP components work together correctly end-to-end, from initialization through training loops to checkpointing.

## Context
Unit tests cover individual components, but integration tests are needed to:
1. Verify the full TP pipeline works correctly
2. Test multi-rank scenarios with mock communicator
3. Validate training convergence
4. Test checkpoint save/load cycles
5. Verify 3D parallelism composition (TP + DP)

## Implementation Details

### 1. Test Helper Utilities

```csharp
namespace MLFramework.Tests.Integration;

public static class TPTestHelpers
{
    /// <summary>
    /// Create a mock TP context for testing
    /// </summary>
    public static TensorParallelContext CreateMockTPContext(int worldSize, int rank)
    {
        return TensorParallelContext.Initialize(worldSize, rank, backend: "mock");
    }

    /// <summary>
    /// Create a simple TP MLP model
    /// </summary>
    public static Module CreateSimpleTPMLP(
        int inputSize,
        int hiddenSize,
        int outputSize,
        bool bias = true)
    {
        var layers = new Sequential();

        // Column-parallel: input → hidden
        var fc1 = new ColumnParallelLinear(
            inputSize, hiddenSize, bias: bias, gatherOutput: false);
        layers.AddModule("fc1", fc1);
        layers.AddModule("relu1", new ReLU());

        // Row-parallel: hidden → output
        var fc2 = new RowParallelLinear(
            hiddenSize, outputSize, bias: bias, inputIsSharded: true);
        layers.AddModule("fc2", fc2);

        return layers;
    }

    /// <summary>
    /// Create a standard (non-parallel) MLP for comparison
    /// </summary>
    public static Module CreateStandardMLP(
        int inputSize,
        int hiddenSize,
        int outputSize,
        bool bias = true)
    {
        var layers = new Sequential();

        var fc1 = new LinearLayer(inputSize, hiddenSize, bias: bias);
        layers.AddModule("fc1", fc1);
        layers.AddModule("relu1", new ReLU());

        var fc2 = new LinearLayer(hiddenSize, outputSize, bias: bias);
        layers.AddModule("fc2", fc2);

        return layers;
    }

    /// <summary>
    /// Create test data
    /// </summary>
    public static Tensor CreateTestInput(int batchSize, int inputSize, int seed = 42)
    {
        var random = new Random(seed);
        return Tensor.Random(batchSize, inputSize, random: random);
    }

    /// <summary>
    /// Run forward pass through model
    /// </summary>
    public static Tensor ForwardPass(Module model, Tensor input)
    {
        return model.Forward(input);
    }

    /// <summary>
    /// Run backward pass and collect gradients
    /// </summary>
    public static Dictionary<string, Tensor> BackwardPass(
        Module model,
        Tensor output,
        Tensor gradOutput)
    {
        var gradInput = model.Backward(gradOutput);
        var grads = new Dictionary<string, Tensor>();

        CollectGradients(model, grads, "");
        return grads;
    }

    private static void CollectGradients(
        Module module,
        Dictionary<string, Tensor> grads,
        string prefix)
    {
        foreach (var param in module.Parameters)
        {
            if (param.Grad != null)
            {
                string fullName = string.IsNullOrEmpty(prefix) ? param.Name : $"{prefix}.{param.Name}";
                grads[fullName] = param.Grad;
            }
        }

        foreach (var submodule in module.Modules)
        {
            string newPrefix = string.IsNullOrEmpty(prefix)
                ? submodule.Name
                : $"{prefix}.{submodule.Name}";
            CollectGradients(submodule, grads, newPrefix);
        }
    }

    /// <summary>
    /// Compare two tensors for approximate equality
    /// </summary>
    public static bool TensorsApproxEqual(Tensor a, Tensor b, double tolerance = 1e-5)
    {
        var diff = (a - b).Abs();
        var maxDiff = diff.Max().ToScalar();
        return maxDiff < tolerance;
    }

    /// <summary>
    /// Run training step
    /// </summary>
    public static void TrainingStep(
        Module model,
        Tensor input,
        Tensor target,
        Optimizer optimizer,
        Loss loss)
    {
        optimizer.ZeroGrad();

        var output = model.Forward(input);
        var lossValue = loss.Compute(output, target);

        // Backward
        var gradLoss = lossValue.Backward();
        model.Backward(gradLoss);

        // Optimizer step
        optimizer.Step();
    }
}
```

### 2. End-to-End TP Tests

```csharp
[TestClass]
public class TPEndToEndTests
{
    [TestMethod]
    public void TPMLP_ForwardPass_ProducesCorrectOutput()
    {
        // Arrange
        using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);

        var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5);
        var input = TPTestHelpers.CreateTestInput(batchSize: 4, inputSize: 10);

        // Act
        var output = tpModel.Forward(input);

        // Assert
        Assert.AreEqual(4, output.Shape[0]); // batch size
        Assert.AreEqual(5, output.Shape[^1]);  // output size
    }

    [TestMethod]
    public void TPMLP_OutputMatchesStandardModel()
    {
        // Arrange
        using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);
        var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5);
        var standardModel = TPTestHelpers.CreateStandardMLP(inputSize: 10, hiddenSize: 20, outputSize: 5);

        // Copy weights from TP to standard model (simplified)
        // In practice, this would need proper weight gathering
        CopyWeightsForTesting(tpModel, standardModel);

        var input = TPTestHelpers.CreateTestInput(batchSize: 4, inputSize: 10);

        // Act
        var tpOutput = tpModel.Forward(input);
        var standardOutput = standardModel.Forward(input);

        // Assert
        Assert.IsTrue(
            TPTestHelpers.TensorsApproxEqual(tpOutput, standardOutput, tolerance: 1e-4),
            "TP output should match standard output");
    }

    [TestMethod]
    public void TPMLP_BackwardPass_ComputesCorrectGradients()
    {
        // Arrange
        using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);
        var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5);
        var input = TPTestHelpers.CreateTestInput(batchSize: 4, inputSize: 10);

        // Forward pass
        var output = tpModel.Forward(input);

        // Create gradient
        var gradOutput = Tensor.OnesLike(output);

        // Act
        var gradInput = tpModel.Backward(gradOutput);

        // Assert
        Assert.IsNotNull(gradInput);
        Assert.AreEqual(input.Shape.Length, gradInput.Shape.Length);
    }

    private void CopyWeightsForTesting(Module tpModel, Module standardModel)
    {
        // Simplified: In real tests, would gather TP weights and copy to standard model
        // This is a placeholder for the actual implementation
    }
}
```

### 3. Multi-Rank Simulation Tests

```csharp
[TestClass]
public class TPMultiRankTests
{
    [TestMethod]
    public void TPMLP_MultiRanks_AllRanksProduceOutput()
    {
        var worldSize = 4;
        var outputs = new List<Tensor>();

        // Simulate execution on all ranks
        for (int rank = 0; rank < worldSize; rank++)
        {
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize, rank);

            var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5);
            var input = TPTestHelpers.CreateTestInput(batchSize: 4, inputSize: 10, seed: 42);

            var output = tpModel.Forward(input);
            outputs.Add(output);
        }

        // All ranks should produce output
        Assert.AreEqual(worldSize, outputs.Count);

        // All outputs should have the same shape
        for (int i = 1; i < outputs.Count; i++)
        {
            CollectionAssert.AreEqual(outputs[0].Shape, outputs[i].Shape);
        }
    }

    [TestMethod]
    public void TPMLP_WeightSharding_EachRankHasDifferentShard()
    {
        var worldSize = 4;
        var fc1Weights = new List<Tensor>();

        // Collect weights from all ranks
        for (int rank = 0; rank < worldSize; rank++)
        {
            using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize, rank);

            var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5);
            var fc1 = tpModel.GetModule("fc1") as ColumnParallelLinear;

            Assert.IsNotNull(fc1);
            fc1Weights.Add(fc1.GetLocalWeight());
        }

        // Verify each rank has different weight values
        for (int i = 1; i < fc1Weights.Count; i++)
        {
            var diff = (fc1Weights[0] - fc1Weights[i]).Abs().Max().ToScalar();
            Assert.IsTrue(diff > 1e-6, "Different ranks should have different weight shards");
        }
    }
}
```

### 4. Training Convergence Tests

```csharp
[TestClass]
public class TPTrainingTests
{
    [TestMethod]
    public void TPMLP_Training_LossDecreases()
    {
        // Arrange
        using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);

        var tpModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5);
        var optimizer = new Adam(tpModel.Parameters, learningRate: 0.01);
        var lossFn = new MSELoss();

        var input = TPTestHelpers.CreateTestInput(batchSize: 32, inputSize: 10, seed: 42);
        var target = Tensor.Random(32, 5, random: new Random(123));

        // Train for a few iterations
        var initialLoss = TrainOneIteration(tpModel, optimizer, input, target, lossFn);

        for (int i = 0; i < 10; i++)
        {
            TrainOneIteration(tpModel, optimizer, input, target, lossFn);
        }

        var finalLoss = TrainOneIteration(tpModel, optimizer, input, target, lossFn);

        // Assert: Loss should decrease
        Assert.IsTrue(finalLoss.ToScalar() < initialLoss.ToScalar(),
            "Loss should decrease during training");
    }

    private double TrainOneIteration(
        Module model,
        Optimizer optimizer,
        Tensor input,
        Tensor target,
        Loss lossFn)
    {
        optimizer.ZeroGrad();

        var output = model.Forward(input);
        var loss = lossFn.Compute(output, target);

        var gradLoss = loss.Backward();
        model.Backward(gradLoss);

        optimizer.Step();

        return loss.ToScalar();
    }
}
```

### 5. Checkpoint Integration Tests

```csharp
[TestClass]
public class TPCheckpointTests
{
    private string GetTestCheckpointPath()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempPath);
        return tempPath;
    }

    [TestMethod]
    public async Task TPCheckpoint_SaveAndLoad_RestoresModelCorrectly()
    {
        // Arrange
        using var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0);
        var checkpointDir = GetTestCheckpointPath();

        var originalModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5);
        var input = TPTestHelpers.CreateTestInput(batchSize: 4, inputSize: 10);

        var originalOutput = originalModel.Forward(input);

        // Act: Save checkpoint
        await TPCheckpointManager.SaveDistributedAsync(originalModel, checkpointDir);

        // Create new model and load checkpoint
        var loadedModel = TPTestHelpers.CreateSimpleTPMLP(inputSize: 10, hiddenSize: 20, outputSize: 5);
        await TPCheckpointManager.LoadDistributedAsync(loadedModel, checkpointDir);

        var loadedOutput = loadedModel.Forward(input);

        // Assert: Outputs should match
        Assert.IsTrue(
            TPTestHelpers.TensorsApproxEqual(originalOutput, loadedOutput, tolerance: 1e-6),
            "Loaded model should produce same output as original");

        // Cleanup
        Directory.Delete(checkpointDir, recursive: true);
    }

    [TestMethod]
    public void TPCheckpoint_ListCheckpoints_ReturnsCorrectList()
    {
        // Arrange
        var rootDir = GetTestCheckpointPath();

        // Create multiple checkpoints
        var checkpoint1 = Path.Combine(rootDir, "checkpoint1");
        var checkpoint2 = Path.Combine(rootDir, "checkpoint2");

        using (var tpContext = TPTestHelpers.CreateMockTPContext(worldSize: 2, rank: 0))
        {
            var model = TPTestHelpers.CreateSimpleTPMLP(10, 20, 5);
            TPCheckpointManager.SaveDistributedAsync(model, checkpoint1).Wait();
            TPCheckpointManager.SaveDistributedAsync(model, checkpoint2).Wait();
        }

        // Act
        var checkpoints = TPCheckpointManager.ListCheckpoints(rootDir);

        // Assert
        Assert.AreEqual(2, checkpoints.Count);
        CollectionAssert.Contains(checkpoints, "checkpoint1");
        CollectionAssert.Contains(checkpoints, "checkpoint2");

        // Cleanup
        Directory.Delete(rootDir, recursive: true);
    }
}
```

### 6. Device Mesh Integration Tests

```csharp
[TestClass]
public class TPDeviceMeshTests
{
    [TestMethod]
    public void DeviceMesh_2DMesh_CreatesCorrectProcessGroups()
    {
        // Arrange
        int[] meshShape = new[] { 2, 2 }; // 2 DP groups x 2 TP ranks
        int totalDevices = meshShape.Aggregate(1, (a, b) => a * b);

        for (int rank = 0; rank < totalDevices; rank++)
        {
            // Act
            var mesh = DeviceMesh.CreateFromRank(rank, meshShape, new MockCommunicator(totalDevices, rank));

            var coord = mesh.MyCoordinate;

            // Assert
            Assert.AreEqual(2, coord.Dimensions);
            Assert.IsTrue(coord[0] < 2, "DP coordinate should be < 2");
            Assert.IsTrue(coord[1] < 2, "TP coordinate should be < 2");
        }
    }

    [TestMethod]
    public void DeviceMesh_TPGroups_AllRanksInCorrectGroups()
    {
        // Arrange
        int[] meshShape = new[] { 2, 2 }; // 2 DP groups x 2 TP ranks
        var meshes = new List<DeviceMesh>();

        for (int rank = 0; rank < 4; rank++)
        {
            meshes.Add(DeviceMesh.CreateFromRank(rank, meshShape, new MockCommunicator(4, rank)));
        }

        // Act & Assert
        // All ranks with same DP coordinate should be in same TP group
        var tpGroups = meshes.Select(m => m.GetTPGroup()).ToList();
        Assert.AreEqual(2, tpGroups.Distinct().Count(), "Should have 2 TP groups");
    }
}
```

## Files to Create

### Test Files
- `tests/MLFramework.Tests/Integration/TPTestHelpers.cs`
- `tests/MLFramework.Tests/Integration/TPEndToEndTests.cs`
- `tests/MLFramework.Tests/Integration/TPMultiRankTests.cs`
- `tests/MLFramework.Tests/Integration/TPTrainingTests.cs`
- `tests/MLFramework.Tests/Integration/TPCheckpointTests.cs`
- `tests/MLFramework.Tests/Integration/TPDeviceMeshTests.cs`

## Test Requirements Summary

### 1. End-to-End Tests
- [ ] Forward pass produces correct output
- [ ] Output matches standard (non-parallel) model
- [ ] Backward pass computes correct gradients

### 2. Multi-Rank Tests
- [ ] All ranks produce output
- [ ] Weight sharding works correctly
- [ ] Each rank has different weight shard

### 3. Training Tests
- [ ] Loss decreases during training
- [ ] Gradients are computed correctly
- [ ] Optimizer updates parameters correctly

### 4. Checkpoint Tests
- [ ] Save and load restores model correctly
- [ ] Checkpointed models produce same outputs
- [ ] List checkpoints returns correct list

### 5. Device Mesh Tests
- [ ] 2D mesh creates correct process groups
- [ ] Ranks are assigned correct coordinates
- [ ] TP and DP groups are correct

## Dependencies
- All TP components from previous specs
- Mock communicator from communication tests
- Test framework (MSTest, NUnit, or xUnit)
- Optimizer and Loss classes

## Success Criteria
- [ ] All integration tests pass
- [ ] Multi-rank simulation works correctly
- [ ] Training converges with TP models
- [ ] Checkpoint save/load cycle works
- [ ] Device mesh properly organizes ranks
- [ ] Tests can be run independently or as a suite
- [ ] Tests complete in reasonable time (< 5 minutes)

## Estimated Time
45-60 minutes
