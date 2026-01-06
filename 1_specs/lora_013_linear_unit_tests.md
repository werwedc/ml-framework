# Spec: LoRALinear Unit Tests

## Overview
Implement comprehensive unit tests for the LoRALinear layer implementation to ensure correctness and reliability.

## Test File Structure

### File: `tests/LoRA/LoRALinearTests.cs`

## Test Categories

### 1. Constructor Tests
```csharp
[TestClass]
public class LoRALinearConstructorTests
{
    [TestMethod]
    public void Constructor_WithValidParameters_CreatesLayer()
    {
        // Arrange
        var baseLayer = new Linear(128, 256);

        // Act
        var lora = new LoRALinear(baseLayer, rank: 8, alpha: 16);

        // Assert
        Assert.IsNotNull(lora);
        Assert.AreEqual(8, lora.Rank);
        Assert.AreEqual(2.0f, lora.ScalingFactor); // 16/8 = 2.0
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void Constructor_WithNullBaseLayer_ThrowsException()
    {
        new LoRALinear(null!, rank: 8, alpha: 16);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void Constructor_WithInvalidRank_ThrowsException()
    {
        var baseLayer = new Linear(128, 256);
        new LoRALinear(baseLayer, rank: 0, alpha: 16);
    }

    [TestMethod]
    public void Constructor_WithDifferentInitalizations_CreatesDifferentWeights()
    {
        var baseLayer1 = new Linear(128, 256);
        var baseLayer2 = new Linear(128, 256);

        var lora1 = new LoRALinear(baseLayer1, rank: 8, alpha: 16,
            initialization: LoRAInitializationStrategy.Standard);
        var lora2 = new LoRALinear(baseLayer2, rank: 8, alpha: 16,
            initialization: LoRAInitializationStrategy.Zero);

        var (a1, b1) = lora1.GetAdapterWeights();
        var (a2, b2) = lora2.GetAdapterWeights();

        // Zero initialization should produce zeros
        var a2Sum = a2.Sum().ToScalar<float>();
        var b2Sum = b2.Sum().ToScalar<float>();

        Assert.AreEqual(0.0f, a2Sum, 1e-6f);
        Assert.AreEqual(0.0f, b2Sum, 1e-6f);
    }
}
```

### 2. Forward Pass Tests
```csharp
[TestClass]
public class LoRALinearForwardTests
{
    [TestMethod]
    public void Forward_WithEnabledAdapter_ProducesExpectedOutput()
    {
        var baseLayer = new Linear(64, 128);
        var lora = new LoRALinear(baseLayer, rank: 4, alpha: 8,
            initialization: LoRAInitializationStrategy.Zero);

        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));

        // Act
        var output = lora.Forward(input);

        // Assert
        Assert.AreEqual(new[] { 32, 128 }, output.Shape);
    }

    [TestMethod]
    public void Forward_WithDisabledAdapter_MatchesBaseOutput()
    {
        var baseLayer = new Linear(64, 128);
        var lora = new LoRALinear(baseLayer, rank: 4, alpha: 8);

        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));

        var baseOutput = baseLayer.Forward(input);
        lora.IsEnabled = false;
        var loraOutput = lora.Forward(input);

        // Outputs should be identical when adapter is disabled
        var diff = Math.Abs(baseOutput - loraOutput).Max().ToScalar<float>();
        Assert.AreEqual(0.0f, diff, 1e-6f);
    }

    [TestMethod]
    public void Forward_WithDifferentBatchSizes_WorksCorrectly()
    {
        var baseLayer = new Linear(64, 128);
        var lora = new LoRALinear(baseLayer, rank: 4, alpha: 8);

        var input1 = Tensor.Random(new[] { 16, 64 }, new Random(42));
        var input2 = Tensor.Random(new[] { 32, 64 }, new Random(42));
        var input3 = Tensor.Random(new[] { 64, 64 }, new Random(42));

        var output1 = lora.Forward(input1);
        var output2 = lora.Forward(input2);
        var output3 = lora.Forward(input3);

        Assert.AreEqual(new[] { 16, 128 }, output1.Shape);
        Assert.AreEqual(new[] { 32, 128 }, output2.Shape);
        Assert.AreEqual(new[] { 64, 128 }, output3.Shape);
    }

    [TestMethod]
    public void Forward_WithDropoutInTrainingMode_AppliesDropout()
    {
        var baseLayer = new Linear(64, 128);
        var lora = new LoRALinear(baseLayer, rank: 4, alpha: 8, dropout: 0.5f);

        lora.IsTrainingMode = true;
        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));

        var output1 = lora.Forward(input);
        var output2 = lora.Forward(input);

        // With dropout, outputs should differ
        var diff = Math.Abs(output1 - output2).Max().ToScalar<float>();
        Assert.IsTrue(diff > 0.0f);
    }
}
```

### 3. Freeze/Unfreeze Tests
```csharp
[TestClass]
public class LoRALinearFreezeTests
{
    [TestMethod]
    public void FreezeBaseLayer_SetsCorrectRequiresGrad()
    {
        var baseLayer = new Linear(64, 128);
        var lora = new LoRALinear(baseLayer, rank: 4, alpha: 8);

        // Act
        lora.FreezeBaseLayer();

        // Assert
        Assert.IsFalse(baseLayer.Weight.RequiresGrad);
    }

    [TestMethod]
    public void UnfreezeBaseLayer_SetsCorrectRequiresGrad()
    {
        var baseLayer = new Linear(64, 128);
        var lora = new LoRALinear(baseLayer, rank: 4, alpha: 8);

        lora.FreezeBaseLayer();
        lora.UnfreezeBaseLayer();

        // Assert
        Assert.IsTrue(baseLayer.Weight.RequiresGrad);
    }

    [TestMethod]
    public void TrainableParameters_WithFrozenBase_ReturnsOnlyAdapterParams()
    {
        var baseLayer = new Linear(64, 128);
        var lora = new LoRALinear(baseLayer, rank: 4, alpha: 8);

        lora.FreezeBaseLayer();
        var trainableParams = lora.TrainableParameters.ToList();

        // Should only have 2 adapter matrices
        Assert.AreEqual(2, trainableParams.Count);
        Assert.IsFalse(baseLayer.Weight.RequiresGrad);
    }
}
```

### 4. Merge/Reset Tests
```csharp
[TestClass]
public class LoRALinearMergeTests
{
    [TestMethod]
    public void MergeAdapter_UpdatesBaseWeights()
    {
        var baseLayer = new Linear(64, 128);
        var lora = new LoRALinear(baseLayer, rank: 4, alpha: 8);

        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));
        var outputBeforeMerge = lora.Forward(input);

        // Act
        lora.MergeAdapter();
        lora.IsEnabled = false;
        var outputAfterMerge = lora.Forward(input);

        // Assert - outputs should be similar
        var diff = Math.Abs(outputBeforeMerge - outputAfterMerge).Max().ToScalar<float>();
        Assert.Less(diff, 1e-4f);
    }

    [TestMethod]
    public void ResetBaseLayer_RestoresOriginalWeights()
    {
        var baseLayer = new Linear(64, 128);
        var lora = new LoRALinear(baseLayer, rank: 4, alpha: 8);

        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));
        var outputBefore = lora.Forward(input);

        lora.MergeAdapter();
        lora.ResetBaseLayer();
        lora.IsEnabled = true;
        var outputAfter = lora.Forward(input);

        // Outputs should be identical after reset
        var diff = Math.Abs(outputBefore - outputAfter).Max().ToScalar<float>();
        Assert.AreEqual(0.0f, diff, 1e-6f);
    }

    [TestMethod]
    [ExpectedException(typeof(InvalidOperationException))]
    public void ResetBaseLayer_WithoutMerge_ThrowsException()
    {
        var baseLayer = new Linear(64, 128);
        var lora = new LoRALinear(baseLayer, rank: 4, alpha: 8);

        lora.ResetBaseLayer();
    }
}
```

### 5. Adapter Weight Tests
```csharp
[TestClass]
public class LoRALinearWeightTests
{
    [TestMethod]
    public void GetAdapterWeights_ReturnsCorrectShapes()
    {
        var baseLayer = new Linear(64, 128);
        var lora = new LoRALinear(baseLayer, rank: 8, alpha: 16);

        var (matrixA, matrixB) = lora.GetAdapterWeights();

        Assert.AreEqual(new[] { 8, 64 }, matrixA.Shape);
        Assert.AreEqual(new[] { 128, 8 }, matrixB.Shape);
    }

    [TestMethod]
    public void SetAdapterWeights_WithCorrectShapes_UpdatesWeights()
    {
        var baseLayer = new Linear(64, 128);
        var lora1 = new LoRALinear(baseLayer, rank: 8, alpha: 16);
        var lora2 = new LoRALinear(new Linear(64, 128), rank: 8, alpha: 16);

        var input = Tensor.Random(new[] { 32, 64 }, new Random(42));

        var (matrixA, matrixB) = lora1.GetAdapterWeights();
        var newMatrixA = matrixA.Clone();
        var newMatrixB = matrixB.Clone();

        // Modify weights
        newMatrixA = newMatrixA.Add(0.1f);
        newMatrixB = newMatrixB.Add(0.1f);

        lora2.SetAdapterWeights(newMatrixA, newMatrixB);

        var output1 = lora1.Forward(input);
        var output2 = lora2.Forward(input);

        // Outputs should differ
        var diff = Math.Abs(output1 - output2).Max().ToScalar<float>();
        Assert.IsTrue(diff > 0.0f);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void SetAdapterWeights_WithNullMatrix_ThrowsException()
    {
        var baseLayer = new Linear(64, 128);
        var lora = new LoRALinear(baseLayer, rank: 8, alpha: 16);

        lora.SetAdapterWeights(null!, null!);
    }

    [TestMethod]
    [ExpectedException(typeof(ArgumentException))]
    public void SetAdapterWeights_WithWrongShapes_ThrowsException()
    {
        var baseLayer = new Linear(64, 128);
        var lora = new LoRALinear(baseLayer, rank: 8, alpha: 16);

        var wrongShapeA = Tensor.Random(new[] { 8, 100 }, new Random(42));
        var wrongShapeB = Tensor.Random(new[] { 128, 8 }, new Random(42));

        lora.SetAdapterWeights(wrongShapeA, wrongShapeB);
    }
}
```

### 6. Performance Tests
```csharp
[TestClass]
public class LoRALinearPerformanceTests
{
    [TestMethod]
    public void Forward_CompareToBaseLayer_MeasureOverhead()
    {
        var baseLayer = new Linear(1024, 2048);
        var lora = new LoRALinear(baseLayer, rank: 8, alpha: 16);

        var input = Tensor.Random(new[] { 32, 1024 }, new Random(42));

        // Warmup
        for (int i = 0; i < 10; i++)
        {
            baseLayer.Forward(input);
            lora.Forward(input);
        }

        // Benchmark base layer
        var baseTimings = new List<long>();
        for (int i = 0; i < 100; i++)
        {
            var sw = Stopwatch.StartNew();
            baseLayer.Forward(input);
            sw.Stop();
            baseTimings.Add(sw.ElapsedMilliseconds);
        }

        // Benchmark LoRA layer
        var loraTimings = new List<long>();
        for (int i = 0; i < 100; i++)
        {
            var sw = Stopwatch.StartNew();
            lora.Forward(input);
            sw.Stop();
            loraTimings.Add(sw.ElapsedMilliseconds);
        }

        var avgBase = baseTimings.Average();
        var avgLoRA = loraTimings.Average();
        var overhead = ((avgLoRA - avgBase) / avgBase) * 100.0;

        // LoRA overhead should be minimal (< 20%)
        Assert.Less(overhead, 20.0);
    }
}
```

## Success Criteria
- All tests pass consistently
- Tests cover all major functionality
- Edge cases are handled correctly
- Performance is within acceptable bounds

## Dependencies
- LoRALinear implementation (from spec 002)
- Testing framework (xUnit/NUnit/etc.)
- Tensor operations (existing)

## Estimated Time
60 minutes

## Notes
- Use parameterized tests for different layer sizes
- Test with random seeds for reproducibility
- Add benchmarks for performance monitoring
- Consider integration with code coverage tools
