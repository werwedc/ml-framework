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

    #region Basic Reproducibility Tests

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

    #endregion

    #region State Checkpointing Tests

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

    #endregion

    #region Scoped Determinism Tests

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

    #endregion

    #region Multi-Device Seeding Tests

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

    #endregion

    #region Deterministic Seed Formula Tests

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

    #endregion

    #region Cross-Run Reproducibility Tests

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

    #endregion

    #region End-to-End Scenario Tests

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

    #endregion

    #region Helper Methods

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

    #endregion
}
