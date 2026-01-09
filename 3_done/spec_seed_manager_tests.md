# Spec: SeedManager Unit Tests

## Overview
Implement comprehensive unit tests for all SeedManager functionality including seeding, state management, deterministic mode, scoped behavior, multi-device seeding, and validation. Tests should follow AAA (Arrange-Act-Assert) pattern and use mocking where appropriate.

## Technical Requirements

### Test File Structure
```
tests/
  MLFramework.Tests/
    Utilities/
      SeedManagerTests.cs
```

### Test Class Definition
```csharp
namespace MLFramework.Tests.Utilities;

using Xunit;
using System;
using MLFramework.Utilities;

public class SeedManagerTests : IDisposable
{
    private SeedManager _seedManager;

    public SeedManagerTests()
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

### 1. Core Seeding Tests

#### Test SetGlobalSeed
```csharp
[Fact]
public void SetGlobalSeed_SetsAllRNGs()
{
    // Arrange
    var seed = 42;

    // Act
    _seedManager.SetGlobalSeed(seed);

    // Assert
    Assert.Equal(seed, _seedManager.CurrentSeed);
    // Additional assertions for individual RNGs when implemented
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
```

#### Test Individual Seed Methods
```csharp
[Fact]
public void SetRandomSeed_SetsSeedCorrectly()
{
    // Arrange
    var seed = 123;

    // Act
    _seedManager.SetRandomSeed(seed);

    // Assert
    // Verify seed is stored and new Random instances use this seed
}

[Fact]
public void SetNumpySeed_SetsSeedCorrectly()
{
    // Arrange
    var seed = 456;

    // Act
    _seedManager.SetNumpySeed(seed);

    // Assert
    // Verify seed is stored for NumPy interop
}

[Fact]
public void SetCudaSeed_SetsSeedCorrectly()
{
    // Arrange
    var seed = 789;

    // Act
    _seedManager.SetCudaSeed(seed);

    // Assert
    // Verify seed is stored for CUDA interop
}
```

### 2. RNG State Serialization Tests

#### Test CaptureRNGState
```csharp
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
```

#### Test RestoreRNGState
```csharp
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
```

#### Test Save/LoadRNGSnapshot
```csharp
[Fact]
public void SaveLoadRNGSnapshot_RoundtripSuccessful()
{
    // Arrange
    _seedManager.SetGlobalSeed(42);
    var snapshot = _seedManager.CaptureRNGState();
    var filePath = Path.Combine(Path.GetTempPath(), "test_rng_snapshot.json");

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
```

### 3. Deterministic Mode Tests

#### Test SetDeterministicMode
```csharp
[Fact]
public void SetDeterministicMode_SetsFlagsCorrectly()
{
    // Arrange
    var flags = DeterministicModeFlags.CudnnDeterministic | DeterministicModeFlags.CublasDeterministic;

    // Act
    _seedManager.SetDeterministicMode(flags);

    // Assert
    Assert.Equal(flags, _seedManager.IsDeterministic);
    Assert.True(_seedManager.IsDeterministicModeEnabled(DeterministicModeFlags.CudnnDeterministic));
    Assert.True(_seedManager.IsDeterministicModeEnabled(DeterministicModeFlags.CublasDeterministic));
}

[Fact]
public void SetDeterministicMode_WithNone_DisablesAll()
{
    // Act
    _seedManager.SetDeterministicMode(DeterministicModeFlags.All);
    _seedManager.SetDeterministicMode(DeterministicModeFlags.None);

    // Assert
    Assert.Equal(DeterministicModeFlags.None, _seedManager.IsDeterministic);
}
```

#### Test Enable/DisableDeterministicMode
```csharp
[Fact]
public void EnableDisableDeterministicMode_UpdatesFlagsCorrectly()
{
    // Act
    _seedManager.EnableDeterministicMode(DeterministicModeFlags.CudnnDeterministic);
    Assert.True(_seedManager.IsDeterministicModeEnabled(DeterministicModeFlags.CudnnDeterministic));

    _seedManager.DisableDeterministicMode(DeterministicModeFlags.CudnnDeterministic);
    Assert.False(_seedManager.IsDeterministicModeEnabled(DeterministicModeFlags.CudnnDeterministic));
}
```

### 4. Scoped Determinism Tests

#### Test WithDeterminism
```csharp
[Fact]
public void WithDeterminism_RestoresPreviousState()
{
    // Arrange
    var initialMode = _seedManager.IsDeterministic;

    // Act
    using (_seedManager.WithDeterminism(true))
    {
        Assert.NotEqual(initialMode, _seedManager.IsDeterministic);
    }

    // Assert
    Assert.Equal(initialMode, _seedManager.IsDeterministic);
}

[Fact]
public void WithDeterminism_EnablesAllInScope()
{
    // Act
    using (_seedManager.WithDeterminism(true))
    {
        // Assert
        Assert.Equal(DeterministicModeFlags.All, _seedManager.IsDeterministic);
    }
}
```

#### Test WithGlobalSeed
```csharp
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
```

#### Test NestedScopes
```csharp
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
```

### 5. Multi-Device Seeding Tests

#### Test SeedAllDevices
```csharp
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
public void GetDeviceSeed_ThrowsOnUnseededDevice()
{
    // Act & Assert
    Assert.Throws<KeyNotFoundException>(() => _seedManager.GetDeviceSeed(0));
}
```

#### Test SeedWorkers
```csharp
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
public void GetWorkerSeed_ThrowsOnUnseededWorker()
{
    // Act & Assert
    Assert.Throws<KeyNotFoundException>(() => _seedManager.GetWorkerSeed(0));
}
```

#### Test GetDeterministicSeed
```csharp
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
```

### 6. Validation and Diagnostics Tests

#### Test RegisterNonDeterministicOperation
```csharp
[Fact]
public void RegisterNonDeterministicOperation_TracksOperations()
{
    // Arrange
    var operation1 = "atomic_add";
    var operation2 = "hashmap_lookup";

    // Act
    _seedManager.RegisterNonDeterministicOperation(operation1);
    _seedManager.RegisterNonDeterministicOperation(operation2);

    // Assert
    var ops = _seedManager.GetNonDeterministicOperations();
    Assert.Contains(operation1, ops);
    Assert.Contains(operation2, ops);
}

[Fact]
public void RegisterNonDeterministicOperation_AvoidsDuplicates()
{
    // Arrange
    var operation = "atomic_add";

    // Act
    _seedManager.RegisterNonDeterministicOperation(operation);
    _seedManager.RegisterNonDeterministicOperation(operation);

    // Assert
    var ops = _seedManager.GetNonDeterministicOperations();
    Assert.Single(ops);
}
```

#### Test ValidateConfiguration
```csharp
[Fact]
public void ValidateConfiguration_WithNoSettings_ReturnsValid()
{
    // Act
    var result = _seedManager.ValidateConfiguration();

    // Assert
    Assert.True(result.IsValid);
}

[Fact]
public void ValidateConfiguration_WithNonDeterministicOps_Warns()
{
    // Arrange
    _seedManager.SetDeterministicMode(DeterministicModeFlags.All);
    _seedManager.RegisterNonDeterministicOperation("atomic_add");

    // Act
    var result = _seedManager.ValidateConfiguration();

    // Assert
    Assert.False(result.IsValid);
    Assert.NotEmpty(result.Errors);
}
```

#### Test GetDiagnosticInfo
```csharp
[Fact]
public void GetDiagnosticInfo_ReturnsComprehensiveInfo()
{
    // Arrange
    _seedManager.SetGlobalSeed(42);
    _seedManager.SetDeterministicMode(DeterministicModeFlags.CudnnDeterministic);
    _seedManager.SeedAllDevices(42, 2);

    // Act
    var info = _seedManager.GetDiagnosticInfo();

    // Assert
    Assert.Equal(42, info.CurrentSeed);
    Assert.Equal(DeterministicModeFlags.CudnnDeterministic, info.DeterministicMode);
    Assert.Equal(2, info.DeviceCount);
    Assert.NotNull(info.PerformanceImpact);
}
```

## Test Requirements

### Test Framework
- Use **xUnit** as the testing framework
- Use **FluentAssertions** for readable assertions (if available)
- Follow AAA (Arrange-Act-Assert) pattern

### Test Naming Convention
- MethodName_Scenario_ExpectedResult
- Use descriptive names that explain the test
- Use Theory with InlineData for parameterized tests

### Test Isolation
- Each test should be independent
- Use constructor for setup
- Use IDisposable for cleanup
- Use temp files for file I/O tests with proper cleanup

### Edge Cases to Test
- Null/empty inputs
- Invalid seed values
- Unseeded devices/workers
- Double-dispose scenarios
- Nested scopes
- Concurrent access (if applicable)

## Success Criteria
1. ✅ All seeding methods are tested
2. ✅ State capture/restore is tested
3. ✅ Deterministic mode is tested
4. ✅ Scoped behavior is tested
5. ✅ Multi-device seeding is tested
6. ✅ Validation and diagnostics are tested
7. ✅ Edge cases are covered
8. ✅ All tests pass consistently

## Dependencies
- spec_seed_manager_core.md - Core functionality to test
- spec_rng_state_serialization.md - State management to test
- spec_deterministic_mode.md - Deterministic mode to test
- spec_scoped_determinism.md - Scoped behavior to test
- spec_multidevice_seeding.md - Multi-device seeding to test
- spec_validation_diagnostics.md - Validation to test

## Notes
- Backend-specific tests (CUDA, NumPy) are placeholders for interop
- Consider adding integration tests for actual reproducibility verification
- Use mocking for backend operations when needed

## Related Specs
- All previous specs - Tests verify all implemented functionality
- spec_reproducibility_integration_tests.md - Complementary integration tests
