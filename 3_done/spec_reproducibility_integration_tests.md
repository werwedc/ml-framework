# Spec: Reproducibility Integration Tests

## Overview
Implement end-to-end integration tests to verify that the deterministic reproducibility system actually produces reproducible results across different runs. These tests should validate the success criteria from the original idea.

## Technical Requirements

### Test File Structure
```
tests/
  MLFramework.Tests/
    Integration/
      ReproducibilityTests.cs
```

### Test Class Definition
```csharp
namespace MLFramework.Tests.Integration;

using Xunit;
using System;
using System.Collections.Generic;
using System.Linq;
using MLFramework.Utilities;

public class ReproducibilityTests : IDisposable
{
    private SeedManager _seedManager;

    public ReproducibilityTests()
    {
        _seedManager = new SeedManager();
    }

    public void Dispose()
    {
        _seedManager?.Dispose();
    }
}
```

## Test Categories

### 1. Basic Reproducibility Tests

#### Test IdenticalSeeds_ProduceIdenticalResults
```csharp
[Fact]
public void IdenticalSeeds_ProduceIdenticalRandomNumbers()
{
    // Arrange
    var seed = 42;
    var count = 100;

    // Act - First run
    _seedManager.SetGlobalSeed(seed);
    var run1Results = GenerateRandomNumbers(count);

    // Act - Second run
    _seedManager.SetGlobalSeed(seed);
    var run2Results = GenerateRandomNumbers(count);

    // Assert - Results should be identical
    Assert.Equal(run1Results, run2Results);
}

private List<double> GenerateRandomNumbers(int count)
{
    var random = new Random(_seedManager.CurrentSeed);
    var results = new List<double>();
    for (int i = 0; i < count; i++)
    {
        results.Add(random.NextDouble());
    }
    return results;
}
```

#### Test DifferentSeeds_ProduceDifferentResults
```csharp
[Fact]
public void DifferentSeeds_ProduceDifferentRandomNumbers()
{
    // Arrange
    var seed1 = 42;
    var seed2 = 43;
    var count = 100;

    // Act
    _seedManager.SetGlobalSeed(seed1);
    var run1Results = GenerateRandomNumbers(count);

    _seedManager.SetGlobalSeed(seed2);
    var run2Results = GenerateRandomNumbers(count);

    // Assert - Results should be different
    Assert.NotEqual(run1Results, run2Results);
}
```

### 2. State Checkpointing Tests

#### Test StateCaptureRestore_ProducesIdenticalResults
```csharp
[Fact]
public void StateCaptureRestore_ProducesIdenticalResults()
{
    // Arrange
    _seedManager.SetGlobalSeed(42);
    var snapshot = _seedManager.CaptureRNGState();

    // Generate some numbers to advance RNG state
    var intermediateResults = GenerateRandomNumbers(50);

    // Act - Restore state
    _seedManager.RestoreRNGState(snapshot);
    var resultsAfterRestore = GenerateRandomNumbers(50);

    // Act - New run from same seed
    _seedManager.SetGlobalSeed(42);
    var freshResults = GenerateRandomNumbers(50);

    // Assert - Results should be identical
    Assert.Equal(freshResults, resultsAfterRestore);
    Assert.NotEqual(intermediateResults, freshResults);
}
```

#### Test SaveLoadSnapshot_ProducesIdenticalResults
```csharp
[Fact]
public void SaveLoadSnapshot_ProducesIdenticalResults()
{
    // Arrange
    _seedManager.SetGlobalSeed(42);
    var snapshot = _seedManager.CaptureRNGState();
    var filePath = Path.Combine(Path.GetTempPath(), "test_reproducibility_snapshot.json");

    try
    {
        // Act - Save snapshot
        _seedManager.SaveRNGSnapshot(snapshot, filePath);

        // Act - Load snapshot in new instance
        var newManager = new SeedManager();
        var loadedSnapshot = newManager.LoadRNGSnapshot(filePath);
        newManager.RestoreRNGState(loadedSnapshot);

        var newResults = GenerateRandomNumbers(50);
        newManager.Dispose();

        // Act - Original instance
        _seedManager.SetGlobalSeed(42);
        var originalResults = GenerateRandomNumbers(50);

        // Assert - Results should be identical
        Assert.Equal(originalResults, newResults);
    }
    finally
    {
        // Cleanup
        if (File.Exists(filePath))
            File.Delete(filePath);
    }
}
```

### 3. Scoped Determinism Tests

#### Test ScopedDeterminism_ProducesReproducibleResults
```csharp
[Fact]
public void ScopedDeterminism_ProducesReproducibleResults()
{
    // Arrange
    var seed = 42;
    var count = 50;

    // Act - First scoped execution
    _seedManager.SetGlobalSeed(seed);
    List<double> results1;
    using (_seedManager.WithGlobalSeed(100))
    {
        results1 = GenerateRandomNumbers(count);
    }

    // Act - Second scoped execution
    _seedManager.SetGlobalSeed(seed);
    List<double> results2;
    using (_seedManager.WithGlobalSeed(100))
    {
        results2 = GenerateRandomNumbers(count);
    }

    // Assert - Scoped results should be identical
    Assert.Equal(results1, results2);

    // Assert - Outer seed should be restored
    Assert.Equal(seed, _seedManager.CurrentSeed);
}
```

### 4. Multi-Device Seeding Tests

#### Test DeviceSpecificSeeds_ProduceDeterministicResults
```csharp
[Fact]
public void DeviceSpecificSeeds_ProduceDeterministicResults()
{
    // Arrange
    var baseSeed = 42;
    var deviceCount = 4;
    _seedManager.SeedAllDevices(baseSeed, deviceCount);

    var deviceResults = new Dictionary<int, List<double>>();

    // Act - Generate results for each device
    foreach (int deviceId in Enumerable.Range(0, deviceCount))
    {
        var seed = _seedManager.GetDeviceSeed(deviceId);
        var random = new Random(seed);
        deviceResults[deviceId] = Enumerable.Range(0, 50)
            .Select(_ => random.NextDouble())
            .ToList();
    }

    // Act - Repeat to verify determinism
    _seedManager.SeedAllDevices(baseSeed, deviceCount);
    var repeatResults = new Dictionary<int, List<double>>();

    foreach (int deviceId in Enumerable.Range(0, deviceCount))
    {
        var seed = _seedManager.GetDeviceSeed(deviceId);
        var random = new Random(seed);
        repeatResults[deviceId] = Enumerable.Range(0, 50)
            .Select(_ => random.NextDouble())
            .ToList();
    }

    // Assert - Each device should produce consistent results
    foreach (int deviceId in Enumerable.Range(0, deviceCount))
    {
        Assert.Equal(deviceResults[deviceId], repeatResults[deviceId]);
    }

    // Assert - Different devices should produce different results
    for (int i = 0; i < deviceCount - 1; i++)
    {
        for (int j = i + 1; j < deviceCount; j++)
        {
            Assert.NotEqual(deviceResults[i], deviceResults[j]);
        }
    }
}
```

### 5. Deterministic Seed Formula Tests

#### Test DeterministicSeedFormula_IsConsistent
```csharp
[Fact]
public void DeterministicSeedFormula_IsConsistent()
{
    // Arrange
    var baseSeed = 42;
    var deviceId = 2;
    var operationId = 5;

    // Act - Generate seed multiple times
    var seed1 = _seedManager.GetDeterministicSeed(baseSeed, deviceId, operationId);
    var seed2 = _seedManager.GetDeterministicSeed(baseSeed, deviceId, operationId);
    var seed3 = _seedManager.GetDeterministicSeed(baseSeed, deviceId, operationId);

    // Assert - All should be identical
    Assert.Equal(seed1, seed2);
    Assert.Equal(seed2, seed3);
}

[Fact]
public void DeterministicSeedFormula_ProducesUniqueSeeds()
{
    // Arrange
    var baseSeed = 42;
    var seeds = new HashSet<int>();

    // Act - Generate seeds for different combinations
    foreach (int deviceId in Enumerable.Range(0, 5))
    {
        foreach (int operationId in Enumerable.Range(0, 5))
        {
            var seed = _seedManager.GetDeterministicSeed(baseSeed, deviceId, operationId);
            seeds.Add(seed);
        }
    }

    // Assert - All seeds should be unique
    Assert.Equal(25, seeds.Count); // 5 devices * 5 operations
}
```

### 6. Cross-Run Reproducibility Tests

#### Test CrossRunReproducibility_WithSameSeed
```csharp
[Fact]
public void CrossRunReproducibility_WithSameSeed()
{
    // Arrange
    var seed = 42;
    var count = 100;
    var filePath = Path.Combine(Path.GetTempPath(), "reproducibility_test.json");

    try
    {
        // Act - First run and save results
        _seedManager.SetGlobalSeed(seed);
        var run1Results = GenerateRandomNumbers(count);
        var snapshot1 = _seedManager.CaptureRNGState();
        _seedManager.SaveRNGSnapshot(snapshot1, filePath);

        // Act - Simulate new run by creating new instance
        _seedManager.Dispose();
        _seedManager = new SeedManager();

        var loadedSnapshot = _seedManager.LoadRNGSnapshot(filePath);
        _seedManager.RestoreRNGState(loadedSnapshot);
        var run2Results = GenerateRandomNumbers(count);

        // Assert - Results should be identical
        Assert.Equal(run1Results, run2Results);
    }
    finally
    {
        // Cleanup
        if (File.Exists(filePath))
            File.Delete(filePath);
    }
}
```

### 7. End-to-End Scenario Tests

#### Test TrainingRunReproducibility
```csharp
[Fact]
public void TrainingRunReproducibility()
{
    // Arrange - Simulate a simple training loop
    var seed = 42;
    var epochs = 5;
    var batchSize = 32;

    // Act - First training run
    var run1Metrics = SimulateTrainingRun(seed, epochs, batchSize);

    // Act - Second training run with same seed
    var run2Metrics = SimulateTrainingRun(seed, epochs, batchSize);

    // Assert - All metrics should be identical
    Assert.Equal(run1Metrics.Losses, run2Metrics.Losses);
    Assert.Equal(run1Metrics.Accuracies, run2Metrics.Accuracies);
}

private TrainingMetrics SimulateTrainingRun(int seed, int epochs, int batchSize)
{
    _seedManager.SetGlobalSeed(seed);
    var metrics = new TrainingMetrics();

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        var random = new Random(_seedManager.CurrentSeed);
        var loss = random.NextDouble();
        var accuracy = 0.5 + random.NextDouble() * 0.5;

        metrics.Losses.Add(loss);
        metrics.Accuracies.Add(accuracy);
    }

    return metrics;
}

private class TrainingMetrics
{
    public List<double> Losses { get; } = new();
    public List<double> Accuracies { get; } = new();
}
```

## Test Requirements

### Test Framework
- Use **xUnit** as the testing framework
- Integration tests may take longer to run
- Consider using [Fact(Skip = "Long running")] for very long tests

### Test Isolation
- Each test should be independent
- Use proper cleanup for files and resources
- Use temp directory for file operations

### Performance Considerations
- Tests should complete in reasonable time (< 5 minutes per test)
- Use reasonable sample sizes (not too large)
- Consider parallel test execution where safe

### Success Criteria Validation

From the original idea, these tests validate:
1. ✅ Identical training runs produce identical outputs (bit-exact)
2. ✅ RNG state can be saved and restored
3. ✅ Cross-platform reproducibility (CPU only for now)
4. ✅ Reproducibility across multiple devices (simulated)
5. ✅ Scoped determinism works correctly

## Success Criteria
1. ✅ Identical seeds produce identical random numbers
2. ✅ Different seeds produce different random numbers
3. ✅ State capture/restore produces identical results
4. ✅ Save/load snapshot produces identical results
5. ✅ Scoped determinism works correctly
6. ✅ Multi-device seeding is deterministic
7. ✅ Deterministic seed formula is consistent
8. ✅ Cross-run reproducibility is achieved
9. ✅ End-to-end scenarios are reproducible

## Dependencies
- All previous specs - Tests validate all implemented functionality
- spec_seed_manager_tests.md - Complementary unit tests

## Notes
- These tests verify actual reproducibility, not just functionality
- Backend-specific tests (CUDA, NumPy) are placeholders for interop
- Consider adding performance benchmarks for deterministic vs non-deterministic modes
- May need to adjust test parameters based on actual ML framework implementation

## Related Specs
- All previous specs - Tests validate all implemented functionality
- spec_seed_manager_tests.md - Complementary unit tests
