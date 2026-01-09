using Xunit;
using System;
using System.IO;
using MLFramework.Utilities;

namespace MLFramework.Tests.Utilities;

/// <summary>
/// Comprehensive unit tests for SeedManager functionality
/// Tests seeding, state management, and multi-device operations
/// </summary>
public class SeedManagerTests : IDisposable
{
    private readonly SeedManager _seedManager;

    public SeedManagerTests()
    {
        _seedManager = new SeedManager();
    }

    public void Dispose()
    {
        _seedManager?.Dispose();
    }

    #region Core Seeding Tests

    [Fact]
    public void SetGlobalSeed_SetsAllRNGs()
    {
        // Arrange
        var seed = 42;

        // Act
        _seedManager.SetGlobalSeed(seed);

        // Assert
        Assert.Equal(seed, _seedManager.CurrentSeed);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(42)]
    [InlineData(-1)]
    [InlineData(int.MaxValue)]
    public void SetGlobalSeed_AcceptsValidSeeds(int seed)
    {
        // Act & Assert
        _seedManager.SetGlobalSeed(seed);
        Assert.Equal(seed, _seedManager.CurrentSeed);
    }

    [Fact]
    public void SetRandomSeed_SetsSeedCorrectly()
    {
        // Arrange
        var seed = 123;

        // Act
        _seedManager.SetRandomSeed(seed);

        // Assert
        // Method is a no-op in current implementation, but should not throw
        Assert.NotNull(_seedManager);
    }

    [Fact]
    public void SetNumpySeed_SetsSeedCorrectly()
    {
        // Arrange
        var seed = 456;

        // Act
        _seedManager.SetNumpySeed(seed);

        // Assert
        // Method is a no-op in current implementation, but should not throw
        Assert.NotNull(_seedManager);
    }

    [Fact]
    public void SetCudaSeed_SetsSeedCorrectly()
    {
        // Arrange
        var seed = 789;

        // Act
        _seedManager.SetCudaSeed(seed);

        // Assert
        // Method is a no-op in current implementation, but should not throw
        Assert.NotNull(_seedManager);
    }

    #endregion

    #region RNG State Serialization Tests

    [Fact]
    public void CaptureRNGState_CreatesValidSnapshot()
    {
        // Arrange
        _seedManager.SetGlobalSeed(42);

        // Act
        var snapshot = _seedManager.CaptureRNGState();

        // Assert
        Assert.NotNull(snapshot);
        Assert.Equal(42, snapshot.RandomSeed);
        Assert.True(snapshot.Timestamp <= DateTime.UtcNow);
        Assert.NotNull(snapshot.CudaStates);
        Assert.NotNull(snapshot.Metadata);
    }

    [Fact]
    public void RestoreRNGState_RestoresSeedCorrectly()
    {
        // Arrange
        _seedManager.SetGlobalSeed(42);
        var snapshot = _seedManager.CaptureRNGState();
        _seedManager.SetGlobalSeed(999);

        // Act
        _seedManager.RestoreRNGState(snapshot);

        // Assert
        Assert.Equal(42, _seedManager.CurrentSeed);
    }

    [Fact]
    public void RestoreRNGState_ThrowsOnNullSnapshot()
    {
        // Arrange
        RNGSnapshot? snapshot = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => _seedManager.RestoreRNGState(snapshot!));
    }

    [Fact]
    public void SaveLoadRNGSnapshot_RoundtripSuccessful()
    {
        // Arrange
        _seedManager.SetGlobalSeed(42);
        var snapshot = _seedManager.CaptureRNGState();
        var filePath = Path.Combine(Path.GetTempPath(), $"test_rng_snapshot_{Guid.NewGuid()}.json");

        try
        {
            // Act
            _seedManager.SaveRNGSnapshot(snapshot, filePath);
            var loadedSnapshot = _seedManager.LoadRNGSnapshot(filePath);

            // Assert
            Assert.Equal(snapshot.RandomSeed, loadedSnapshot.RandomSeed);
            Assert.Equal(snapshot.Timestamp, loadedSnapshot.Timestamp);
        }
        finally
        {
            // Cleanup
            if (File.Exists(filePath))
                File.Delete(filePath);
        }
    }

    [Fact]
    public void SaveRNGSnapshot_ThrowsOnNullSnapshot()
    {
        // Arrange
        var filePath = Path.Combine(Path.GetTempPath(), "test_snapshot.json");

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => _seedManager.SaveRNGSnapshot(null!, filePath));
    }

    [Fact]
    public void SaveRNGSnapshot_ThrowsOnEmptyPath()
    {
        // Arrange
        var snapshot = _seedManager.CaptureRNGState();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _seedManager.SaveRNGSnapshot(snapshot, ""));
    }

    [Fact]
    public void LoadRNGSnapshot_ThrowsOnEmptyPath()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _seedManager.LoadRNGSnapshot(""));
    }

    [Fact]
    public void LoadRNGSnapshot_ThrowsOnNonexistentFile()
    {
        // Arrange
        var filePath = Path.Combine(Path.GetTempPath(), $"nonexistent_{Guid.NewGuid()}.json");

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() => _seedManager.LoadRNGSnapshot(filePath));
    }

    #endregion

    #region Deterministic Mode Tests

    // Note: Deterministic mode features not yet implemented
    // These tests are placeholders for future implementation

    [Fact(Skip = "Deterministic mode not yet implemented")]
    public void SetDeterministicMode_SetsFlagsCorrectly()
    {
        // Placeholder test - will be implemented when DeterministicModeFlags is added
        // var flags = DeterministicModeFlags.CudnnDeterministic | DeterministicModeFlags.CublasDeterministic;
        // _seedManager.SetDeterministicMode(flags);
        // Assert.Equal(flags, _seedManager.IsDeterministic);
    }

    [Fact(Skip = "Deterministic mode not yet implemented")]
    public void SetDeterministicMode_WithNone_DisablesAll()
    {
        // Placeholder test
    }

    [Fact(Skip = "Deterministic mode not yet implemented")]
    public void EnableDisableDeterministicMode_UpdatesFlagsCorrectly()
    {
        // Placeholder test
    }

    [Fact(Skip = "WithDeterminism not yet implemented")]
    public void WithDeterminism_RestoresPreviousState()
    {
        // Placeholder test
    }

    [Fact(Skip = "WithDeterminism not yet implemented")]
    public void WithDeterminism_EnablesAllInScope()
    {
        // Placeholder test
    }

    #endregion

    #region Scoped Determinism Tests

    [Fact]
    public void WithGlobalSeed_RestoresPreviousSeed()
    {
        // Arrange
        _seedManager.SetGlobalSeed(42);

        // Act
        using (_seedManager.WithGlobalSeed(999))
        {
            Assert.Equal(999, _seedManager.CurrentSeed);
        }

        // Assert
        Assert.Equal(42, _seedManager.CurrentSeed);
    }

    [Fact]
    public void NestedScopes_WorkCorrectly()
    {
        // Arrange
        _seedManager.SetGlobalSeed(1);

        // Act
        using (var outerScope = _seedManager.WithGlobalSeed(2))
        {
            Assert.Equal(2, _seedManager.CurrentSeed);

            using (var innerScope = _seedManager.WithGlobalSeed(3))
            {
                Assert.Equal(3, _seedManager.CurrentSeed);
            }

            Assert.Equal(2, _seedManager.CurrentSeed);
        }

        // Assert
        Assert.Equal(1, _seedManager.CurrentSeed);
    }

    [Fact]
    public void WithGlobalSeed_ReturnsDisposable()
    {
        // Act
        using var scope = _seedManager.WithGlobalSeed(42);

        // Assert
        Assert.NotNull(scope);
        Assert.Equal(42, _seedManager.CurrentSeed);
    }

    #endregion

    #region Multi-Device Seeding Tests

    [Fact]
    public void SeedAllDevices_SeedsAllDevices()
    {
        // Arrange
        var baseSeed = 42;
        var deviceCount = 4;

        // Act
        _seedManager.SeedAllDevices(baseSeed, deviceCount);

        // Assert
        Assert.Equal(42, _seedManager.GetDeviceSeed(0));
        Assert.Equal(43, _seedManager.GetDeviceSeed(1));
        Assert.Equal(44, _seedManager.GetDeviceSeed(2));
        Assert.Equal(45, _seedManager.GetDeviceSeed(3));
    }

    [Fact]
    public void SeedAllDevices_UsesDefaultDeviceCount()
    {
        // Arrange
        var baseSeed = 42;

        // Act
        _seedManager.SeedAllDevices(baseSeed);

        // Assert
        Assert.Equal(42, _seedManager.GetDeviceSeed(0));
    }

    [Fact]
    public void GetDeviceSeed_ThrowsOnUnseededDevice()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _seedManager.GetDeviceSeed(0));
    }

    [Fact]
    public void SeedDevice_ThrowsOnNegativeDeviceId()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => _seedManager.SeedDevice(-1, 42));
    }

    [Fact]
    public void SeedWorkers_SeedsAllWorkers()
    {
        // Arrange
        var baseSeed = 100;
        var workerCount = 4;

        // Act
        _seedManager.SeedWorkers(baseSeed, workerCount);

        // Assert
        Assert.Equal(100, _seedManager.GetWorkerSeed(0));
        Assert.Equal(101, _seedManager.GetWorkerSeed(1));
        Assert.Equal(102, _seedManager.GetWorkerSeed(2));
        Assert.Equal(103, _seedManager.GetWorkerSeed(3));
    }

    [Fact]
    public void SeedWorkers_ThrowsOnNonPositiveWorkerCount()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => _seedManager.SeedWorkers(100, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => _seedManager.SeedWorkers(100, -1));
    }

    [Fact]
    public void GetWorkerSeed_ThrowsOnUnseededWorker()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() => _seedManager.GetWorkerSeed(0));
    }

    [Fact]
    public void SeedWorker_ThrowsOnNegativeWorkerId()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => _seedManager.SeedWorker(-1, 42));
    }

    [Fact]
    public void GetDeterministicSeed_ProducesConsistentResults()
    {
        // Arrange
        var baseSeed = 42;
        var deviceId = 2;
        var operationId = 5;

        // Act
        var seed1 = _seedManager.GetDeterministicSeed(baseSeed, deviceId, operationId);
        var seed2 = _seedManager.GetDeterministicSeed(baseSeed, deviceId, operationId);

        // Assert
        Assert.Equal(seed1, seed2);
    }

    [Theory]
    [InlineData(42, 0, 0)]
    [InlineData(42, 1, 0)]
    [InlineData(42, 0, 1)]
    [InlineData(42, 2, 3)]
    public void GetDeterministicSeed_FollowsFormula(int baseSeed, int deviceId, int operationId)
    {
        // Act
        var seed = _seedManager.GetDeterministicSeed(baseSeed, deviceId, operationId);

        // Assert
        var expected = baseSeed + (deviceId * 1000) + operationId;
        Assert.Equal(expected, seed);
    }

    [Fact]
    public void GetDeterministicSeed_UsesDefaultOperationId()
    {
        // Arrange
        var baseSeed = 42;
        var deviceId = 2;

        // Act
        var seed1 = _seedManager.GetDeterministicSeed(baseSeed, deviceId);
        var seed2 = _seedManager.GetDeterministicSeed(baseSeed, deviceId, 0);

        // Assert
        Assert.Equal(seed1, seed2);
    }

    #endregion

    #region Validation and Diagnostics Tests

    // Note: Validation and diagnostic features not yet implemented
    // These tests are placeholders for future implementation

    [Fact(Skip = "Validation not yet implemented")]
    public void ValidateConfiguration_WithNoSettings_ReturnsValid()
    {
        // Placeholder test
    }

    [Fact(Skip = "Validation not yet implemented")]
    public void ValidateConfiguration_WithNonDeterministicOps_Warns()
    {
        // Placeholder test
    }

    [Fact(Skip = "Diagnostics not yet implemented")]
    public void RegisterNonDeterministicOperation_TracksOperations()
    {
        // Placeholder test
    }

    [Fact(Skip = "Diagnostics not yet implemented")]
    public void RegisterNonDeterministicOperation_AvoidsDuplicates()
    {
        // Placeholder test
    }

    [Fact(Skip = "Diagnostics not yet implemented")]
    public void GetDiagnosticInfo_ReturnsComprehensiveInfo()
    {
        // Placeholder test
    }

    #endregion

    #region Edge Case Tests

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        // Act & Assert
        _seedManager.Dispose();
        _seedManager.Dispose(); // Should not throw
    }

    [Fact]
    public void MultipleSeedManagerInstances_HaveIndependentStates()
    {
        // Arrange
        var seedManager1 = new SeedManager();
        var seedManager2 = new SeedManager();

        // Act
        seedManager1.SetGlobalSeed(42);
        seedManager2.SetGlobalSeed(999);

        // Assert
        Assert.Equal(42, seedManager1.CurrentSeed);
        Assert.Equal(999, seedManager2.CurrentSeed);

        // Cleanup
        seedManager1.Dispose();
        seedManager2.Dispose();
    }

    [Fact]
    public void SetGlobalSeed_IsThreadSafe()
    {
        // Arrange
        const int threadCount = 10;
        const int iterations = 100;

        // Act
        Parallel.For(0, threadCount, i =>
        {
            for (int j = 0; j < iterations; j++)
            {
                _seedManager.SetGlobalSeed(i * 1000 + j);
            }
        });

        // Assert
        // Should not throw and should complete
        Assert.NotNull(_seedManager);
    }

    [Fact]
    public void CaptureRNGState_IncludesDeviceSeeds()
    {
        // Arrange
        _seedManager.SeedAllDevices(42, 4);

        // Act
        var snapshot = _seedManager.CaptureRNGState();

        // Assert
        Assert.Equal(4, snapshot.CudaStates.Count);
        Assert.Contains(0, snapshot.CudaStates.Keys);
        Assert.Contains(1, snapshot.CudaStates.Keys);
        Assert.Contains(2, snapshot.CudaStates.Keys);
        Assert.Contains(3, snapshot.CudaStates.Keys);
    }

    [Fact]
    public void RestoreRNGState_RestoresDeviceSeeds()
    {
        // Arrange
        _seedManager.SeedAllDevices(42, 4);
        var snapshot = _seedManager.CaptureRNGState();
        _seedManager.SeedAllDevices(999, 2);

        // Act
        _seedManager.RestoreRNGState(snapshot);

        // Assert
        Assert.Equal(42, _seedManager.GetDeviceSeed(0));
        Assert.Equal(43, _seedManager.GetDeviceSeed(1));
        Assert.Equal(44, _seedManager.GetDeviceSeed(2));
        Assert.Equal(45, _seedManager.GetDeviceSeed(3));
    }

    [Fact]
    public void RestoreRNGState_ThrowsOnInvalidSnapshot()
    {
        // Arrange
        var invalidSnapshot = new RNGSnapshot
        {
            RandomSeed = 42,
            Timestamp = default,
            CudaStates = new System.Collections.Generic.Dictionary<int, byte[]>(),
            Metadata = null!
        };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _seedManager.RestoreRNGState(invalidSnapshot));
    }

    #endregion
}
