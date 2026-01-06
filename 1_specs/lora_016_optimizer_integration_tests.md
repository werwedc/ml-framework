# Spec: Integration Tests with Optimizer

## Overview
Implement integration tests to verify LoRA works correctly with the optimizer, including training loops, gradient computation, and parameter updates.

## Test File Structure

### File: `tests/LoRA/LoRAOptimizerIntegrationTests.cs`

## Test Categories

### 1. Optimizer Integration Tests
```csharp
[TestClass]
public class LoRAOptimizerIntegrationTests
{
    [TestMethod]
    public void Optimizer_WithLoRA_AdapterOnly_UpdatesAdapterParameters()
    {
        // Arrange
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);
        model.FreezeLoRABaseLayers();

        var paramManager = new ParameterManager(model);
        var trainableParams = paramManager.GetTrainableParameters().ToList();
        var optimizer = new AdamOptimizer(trainableParams, lr: 1e-3f);

        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));
        var target = Tensor.Random(new[] { 32, 128 }, new Random(42));

        // Get initial adapter weights
        var adapters = model.GetLoRAAdapters().ToList();
        var (initialA, initialB) = adapters[0].GetAdapterWeights();

        // Act - training step
        var output = model.Forward(input);
        var loss = Tensor.Mean((output - target).Pow(2));

        loss.Backward();
        optimizer.Step();

        // Assert - adapter weights should have changed
        var (updatedA, updatedB) = adapters[0].GetAdapterWeights();
        var diffA = Math.Abs(initialA! - updatedA!).Sum().ToScalar<float>();
        var diffB = Math.Abs(initialB! - updatedB!).Sum().ToScalar<float>();

        Assert.IsTrue(diffA > 0.0f);
        Assert.IsTrue(diffB > 0.0f);
    }

    [TestMethod]
    public void Optimizer_WithFrozenBase_DoesNotUpdateBaseParameters()
    {
        // Arrange
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);
        model.FreezeLoRABaseLayers();

        var paramManager = new ParameterManager(model);
        var trainableParams = paramManager.GetTrainableParameters().ToList();

        // Get initial base weights
        var baseLayer = model.GetLoRAAdapters().First().BaseLayer as Linear;
        var initialWeights = baseLayer!.Weight.Clone();

        var optimizer = new AdamOptimizer(trainableParams, lr: 1e-3f);
        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));
        var target = Tensor.Random(new[] { 32, 128 }, new Random(42));

        // Act - training step
        for (int i = 0; i < 10; i++)
        {
            var output = model.Forward(input);
            var loss = Tensor.Mean((output - target).Pow(2));
            loss.Backward();
            optimizer.Step();
        }

        // Assert - base weights should not have changed
        var diff = Math.Abs(initialWeights - baseLayer.Weight).Sum().ToScalar<float>();
        Assert.AreEqual(0.0f, diff, 1e-6f);
    }

    [TestMethod]
    public void Optimizer_WithDifferentLearningRates_UsesCorrectLRS()
    {
        // Arrange
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);
        model.FreezeLoRABaseLayers();

        var paramManager = new ParameterManager(model);
        var paramGroups = paramManager.GetParameterGroups(baseLayerLR: 1e-4f, adapterLR: 1e-3f);

        var optimizer = new AdamOptimizer(paramGroups);
        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));
        var target = Tensor.Random(new[] { 32, 128 }, new Random(42));

        // Get initial adapter weights
        var adapters = model.GetLoRAAdapters().ToList();
        var (initialA, _) = adapters[0].GetAdapterWeights();

        // Act - training step
        var output = model.Forward(input);
        var loss = Tensor.Mean((output - target).Pow(2));

        loss.Backward();
        optimizer.Step();

        // Assert - verify learning rate was applied correctly
        var (updatedA, _) = adapters[0].GetAdapterWeights();
        var gradA = initialA!.Grad;
        var expectedChange = gradA.Mul(1e-3f); // Using adapter LR

        var actualChange = updatedA!.Sub(initialA);
        var diff = Math.Abs(expectedChange - actualChange).Max().ToScalar<float>();

        Assert.Less(diff, 1e-4f);
    }
}
```

### 2. Training Loop Tests
```csharp
[TestClass]
public class LoRATrainingLoopTests
{
    [TestMethod]
    public void TrainingLoop_WithLoRA_ReducesLoss()
    {
        // Arrange
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);
        model.FreezeLoRABaseLayers();

        var paramManager = new ParameterManager(model);
        var trainableParams = paramManager.GetTrainableParameters().ToList();
        var optimizer = new AdamOptimizer(trainableParams, lr: 1e-3f);

        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));
        var target = Tensor.Random(new[] { 32, 128 }, new Random(42));

        // Get initial loss
        var initialOutput = model.Forward(input);
        var initialLoss = Tensor.Mean((initialOutput - target).Pow(2)).ToScalar<float>();

        // Act - training loop
        for (int epoch = 0; epoch < 100; epoch++)
        {
            var output = model.Forward(input);
            var loss = Tensor.Mean((output - target).Pow(2));
            loss.Backward();
            optimizer.Step();
        }

        // Assert - loss should have decreased
        var finalOutput = model.Forward(input);
        var finalLoss = Tensor.Mean((finalOutput - target).Pow(2)).ToScalar<float>();

        Assert.Less(finalLoss, initialLoss);
    }

    [TestMethod]
    public void TrainingLoop_WithUnfrozenBase_UpdatesBoth()
    {
        // Arrange
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);
        // Don't freeze base layers

        var paramManager = new ParameterManager(model);
        var trainableParams = paramManager.GetTrainableParameters().ToList();
        var optimizer = new AdamOptimizer(trainableParams, lr: 1e-3f);

        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));
        var target = Tensor.Random(new[] { 32, 128 }, new Random(42));

        // Get initial weights
        var baseLayer = model.GetLoRAAdapters().First().BaseLayer as Linear;
        var initialBaseWeights = baseLayer!.Weight.Clone();
        var (initialAdapterA, _) = model.GetLoRAAdapters().First().GetAdapterWeights();

        // Act - training step
        for (int epoch = 0; epoch < 10; epoch++)
        {
            var output = model.Forward(input);
            var loss = Tensor.Mean((output - target).Pow(2));
            loss.Backward();
            optimizer.Step();
        }

        // Assert - both base and adapter weights should have changed
        var baseDiff = Math.Abs(initialBaseWeights - baseLayer.Weight).Sum().ToScalar<float>();
        var adapterDiff = Math.Abs(initialAdapterA! -
            model.GetLoRAAdapters().First().GetAdapterWeights().MatrixA!).Sum().ToScalar<float>();

        Assert.IsTrue(baseDiff > 0.0f);
        Assert.IsTrue(adapterDiff > 0.0f);
    }
}
```

### 3. Gradient Computation Tests
```csharp
[TestClass]
public class LoRAGradientTests
{
    [TestMethod]
    public void Gradient_WithLoRA_ComputesCorrectGradients()
    {
        // Arrange
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);
        model.FreezeLoRABaseLayers();

        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));
        var target = Tensor.Random(new[] { 32, 128 }, new Random(42));

        // Act - forward and backward pass
        var output = model.Forward(input);
        var loss = Tensor.Mean((output - target).Pow(2));
        loss.Backward();

        // Assert - adapter parameters should have gradients
        var adapters = model.GetLoRAAdapters().ToList();
        foreach (var adapter in adapters)
        {
            var (matrixA, matrixB) = adapter.GetAdapterWeights();

            Assert.IsNotNull(matrixA!.Grad);
            Assert.IsNotNull(matrixB!.Grad);
            Assert.IsTrue(matrixA.Grad.Abs().Sum().ToScalar<float>() > 0.0f);
            Assert.IsTrue(matrixB.Grad.Abs().Sum().ToScalar<float>() > 0.0f);
        }
    }

    [TestMethod]
    public void Gradient_WithFrozenBase_NoGradientsOnBase()
    {
        // Arrange
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);
        model.FreezeLoRABaseLayers();

        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));
        var target = Tensor.Random(new[] { 32, 128 }, new Random(42));

        // Act - forward and backward pass
        var output = model.Forward(input);
        var loss = Tensor.Mean((output - target).Pow(2));
        loss.Backward();

        // Assert - base parameters should not have gradients
        var baseLayer = model.GetLoRAAdapters().First().BaseLayer as Linear;
        var baseWeights = baseLayer!.Weight;

        Assert.IsTrue(baseWeights.Grad == null ||
            baseWeights.Grad.Abs().Sum().ToScalar<float>() < 1e-10f);
    }
}
```

### 4. Parameter Counting Tests
```csharp
[TestClass]
public class LoRAParameterCountingTests
{
    [TestMethod]
    public void ParameterCounting_WithLoRA_ShowsReduction()
    {
        // Arrange
        var model = CreateTestModel();

        // Count parameters before LoRA
        var paramManagerBefore = new ParameterManager(model);
        var statsBefore = paramManagerBefore.GetParameterStats();

        model.ApplyLoRA(rank: 8, alpha: 16);
        model.FreezeLoRABaseLayers();

        // Count parameters after LoRA
        var paramManagerAfter = new ParameterManager(model);
        var statsAfter = paramManagerAfter.GetParameterStats();

        // Assert - trainable parameters should be much fewer
        Assert.Less(statsAfter.TrainableParameters, statsBefore.TrainableParameters);
        Assert.IsTrue(statsAfter.ReductionPercentage > 80.0); // >80% reduction
    }

    [TestMethod]
    public void ParameterGrouping_ByLearningRate_SeparatesCorrectly()
    {
        // Arrange
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);

        var paramManager = new ParameterManager(model);
        var paramGroups = paramManager.GetParameterGroups(baseLayerLR: 1e-4f, adapterLR: 1e-3f);

        // Assert - should have two groups
        Assert.IsTrue(paramGroups.ContainsKey(1e-4f));
        Assert.IsTrue(paramGroups.ContainsKey(1e-3f));

        // Adapter parameters should be in high LR group
        var highLRParams = paramGroups[1e-3f];
        Assert.IsTrue(highLRParams.Count > 0);
    }
}
```

### 5. End-to-End Training Tests
```csharp
[TestClass]
public class LoRAEndToEndTests
{
    [TestMethod]
    public void EndToEnd_TrainingWithLoRA_ImprovesPerformance()
    {
        // Arrange
        var model = CreateTestModel();
        model.ApplyLoRA(rank: 8, alpha: 16);
        model.FreezeLoRABaseLayers();

        var paramManager = new ParameterManager(model);
        var trainableParams = paramManager.GetTrainableParameters().ToList();
        var optimizer = new AdamOptimizer(trainableParams, lr: 1e-3f);

        // Create synthetic training data
        var trainingData = new List<(ITensor Input, ITensor Target)>();
        for (int i = 0; i < 100; i++)
        {
            var input = Tensor.Random(new[] { 32, 64 }, new Random(i));
            var target = input.Linear(new Linear(64, 128));
            trainingData.Add((input, target));
        }

        // Act - training loop
        var losses = new List<float>();
        foreach (var (input, target) in trainingData)
        {
            var output = model.Forward(input);
            var loss = Tensor.Mean((output - target).Pow(2));
            loss.Backward();
            optimizer.Step();

            losses.Add(loss.ToScalar<float>());
        }

        // Assert - loss should trend downward
        var initialLoss = losses.Take(10).Average();
        var finalLoss = losses.Skip(losses.Count - 10).Average();
        Assert.Less(finalLoss, initialLoss);

        // Loss should decrease overall
        Assert.IsTrue(finalLoss < initialLoss * 0.9); // At least 10% improvement
    }

    [TestMethod]
    public void EndToEnd_SaveLoadAdapter_PreservesTraining()
    {
        // Arrange
        const string TestRegistryPath = "./test_adapters";
        if (Directory.Exists(TestRegistryPath))
            Directory.Delete(TestRegistryPath, recursive: true);

        var model1 = CreateTestModel();
        model1.ApplyLoRA(rank: 8, alpha: 16);
        model1.FreezeLoRABaseLayers();

        // Train for a few steps
        var paramManager = new ParameterManager(model1);
        var optimizer = new AdamOptimizer(paramManager.GetTrainableParameters().ToList(), lr: 1e-3f);

        for (int i = 0; i < 10; i++)
        {
            var input = Tensor.Random(new[] { 32, 64 }, new Random(i));
            var target = Tensor.Random(new[] { 32, 128 }, new Random(i + 1));
            var output = model1.Forward(input);
            var loss = Tensor.Mean((output - target).Pow(2));
            loss.Backward();
            optimizer.Step();
        }

        // Save adapter
        var registry = new LoRAAdapterRegistry(TestRegistryPath);
        registry.SetModel(model1);
        registry.SaveAdapter("trained_adapter");

        // Load into new model
        var model2 = CreateTestModel();
        model2.ApplyLoRA(rank: 8, alpha: 16);
        registry.SetModel(model2);
        registry.LoadAdapter("trained_adapter");

        // Act - compare outputs
        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));
        var output1 = model1.Forward(input);
        var output2 = model2.Forward(input);

        // Assert - outputs should be similar
        var diff = Math.Abs(output1 - output2).Max().ToScalar<float>();
        Assert.Less(diff, 1e-4f);

        // Cleanup
        Directory.Delete(TestRegistryPath, recursive: true);
    }
}
```

## Helper Methods
```csharp
private IModule CreateTestModel()
{
    var layer1 = new Linear(64, 128);
    var layer2 = new Linear(128, 256);
    var layer3 = new Linear(256, 128);

    return new SimpleSequential(new[] { layer1, layer2, layer3 });
}

private class SimpleSequential : IModule, IHasSubmodules
{
    private readonly List<IModule> _layers;

    public SimpleSequential(IModule[] layers)
    {
        _layers = layers.ToList();
    }

    public ITensor Forward(ITensor input)
    {
        var output = input;
        foreach (var layer in _layers)
        {
            output = layer.Forward(output);
            output = output.Relu();
        }
        return output;
    }

    public IEnumerable<(string Name, IModule Module)> NamedChildren()
    {
        return _layers.Select((layer, i) => ($"layer{i}", layer));
    }

    public void SetModule(string name, IModule module)
    {
        var index = int.Parse(name.Replace("layer", ""));
        _layers[index] = module;
    }
}
```

## Success Criteria
- Optimizer correctly updates adapter parameters
- Base parameters remain frozen when specified
- Gradients are computed correctly
- Training loops reduce loss
- Save/load preserves training progress
- All tests pass consistently

## Dependencies
- LoRA adapters and utilities (from specs 001-012)
- AdamOptimizer or similar optimizer (existing)
- ParameterManager (from spec 006)
- Testing framework (xUnit/NUnit/etc.)

## Estimated Time
60 minutes

## Notes
- Use small models for fast test execution
- Test with different optimizers if available
- Verify gradient flow through LoRA layers
- Test with different batch sizes
- Ensure memory usage is reasonable
