# Spec: Distributed Checkpointing - Unit Tests

## Overview
Implement comprehensive unit tests for all checkpointing components to ensure correctness, reliability, and maintainability.

## Scope
- 45-60 minutes coding time
- Focus on test coverage and edge cases
- Target: `tests/MLFramework/Checkpointing/`

## Test Structure

### 1. Storage Tests (`StorageTests.cs`)
```csharp
[TestFixture]
public class LocalFileSystemStorageTests
{
    private string _testDirectory;
    private LocalFileSystemStorage _storage;

    [SetUp]
    public void SetUp()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"checkpoint_test_{Guid.NewGuid():N}");
        _storage = new LocalFileSystemStorage(_testDirectory);
    }

    [TearDown]
    public void TearDown()
    {
        if (Directory.Exists(_testDirectory))
        {
            Directory.Delete(_testDirectory, recursive: true);
        }
    }

    [Test]
    public async Task WriteAsync_CreatesDirectory_IfNotExists()
    {
        // Arrange
        var data = Encoding.UTF8.GetBytes("test data");
        var path = "subdir/file.txt";

        // Act
        await _storage.WriteAsync(path, data);

        // Assert
        var fullPath = Path.Combine(_testDirectory, path);
        Assert.That(File.Exists(fullPath), Is.True);
    }

    [Test]
    public async Task WriteAndReadAsync_Roundtrip_Success()
    {
        // Arrange
        var expectedData = Encoding.UTF8.GetBytes("test data");
        var path = "test.txt";

        // Act
        await _storage.WriteAsync(path, expectedData);
        var actualData = await _storage.ReadAsync(path);

        // Assert
        Assert.That(actualData, Is.EqualTo(expectedData));
    }

    [Test]
    public async Task ExistsAsync_ReturnsTrue_IfFileExists()
    {
        // Arrange
        var data = Encoding.UTF8.GetBytes("test data");
        var path = "test.txt";
        await _storage.WriteAsync(path, data);

        // Act
        var exists = await _storage.ExistsAsync(path);

        // Assert
        Assert.That(exists, Is.True);
    }

    [Test]
    public async Task ExistsAsync_ReturnsFalse_IfFileNotExists()
    {
        // Act
        var exists = await _storage.ExistsAsync("nonexistent.txt");

        // Assert
        Assert.That(exists, Is.False);
    }

    [Test]
    public async Task DeleteAsync_RemovesFile()
    {
        // Arrange
        var data = Encoding.UTF8.GetBytes("test data");
        var path = "test.txt";
        await _storage.WriteAsync(path, data);

        // Act
        await _storage.DeleteAsync(path);
        var exists = await _storage.ExistsAsync(path);

        // Assert
        Assert.That(exists, Is.False);
    }

    [Test]
    public async Task GetMetadataAsync_ReturnsCorrectMetadata()
    {
        // Arrange
        var data = Encoding.UTF8.GetBytes("test data");
        var path = "test.txt";
        await _storage.WriteAsync(path, data);

        // Act
        var metadata = await _storage.GetMetadataAsync(path);

        // Assert
        Assert.That(metadata.Size, Is.EqualTo(data.Length));
    }
}
```

### 2. Metadata Tests (`MetadataTests.cs`)
```csharp
[TestFixture]
public class MetadataTests
{
    [Test]
    public void SerializeAndDeserialize_Roundtrip_Success()
    {
        // Arrange
        var expectedMetadata = CreateTestMetadata();

        // Act
        var json = MetadataSerializer.Serialize(expectedMetadata);
        var actualMetadata = MetadataSerializer.Deserialize(json);

        // Assert
        Assert.That(actualMetadata.Version, Is.EqualTo(expectedMetadata.Version));
        Assert.That(actualMetadata.WorldSize, Is.EqualTo(expectedMetadata.WorldSize));
    }

    [Test]
    public void Validate_WithValidMetadata_ReturnsValid()
    {
        // Arrange
        var metadata = CreateTestMetadata();

        // Act
        var result = MetadataValidator.Validate(metadata);

        // Assert
        Assert.That(result.IsValid, Is.True);
        Assert.That(result.Errors, Is.Empty);
    }

    [Test]
    public void Validate_WithMissingVersion_ReturnsInvalid()
    {
        // Arrange
        var metadata = CreateTestMetadata();
        metadata.Version = null;

        // Act
        var result = MetadataValidator.Validate(metadata);

        // Assert
        Assert.That(result.IsValid, Is.False);
        Assert.That(result.Errors, Has.Some.Contains("Version"));
    }

    [Test]
    public void Validate_WithShardCountMismatch_ReturnsInvalid()
    {
        // Arrange
        var metadata = CreateTestMetadata();
        metadata.Sharding.ShardCount = 4;
        metadata.Shards = new List<ShardMetadata>(new ShardMetadata[2]);

        // Act
        var result = MetadataValidator.Validate(metadata);

        // Assert
        Assert.That(result.IsValid, Is.False);
        Assert.That(result.Errors, Has.Some.Contains("Shard count"));
    }

    private static CheckpointMetadata CreateTestMetadata()
    {
        return new CheckpointMetadata
        {
            Version = "1.0.0",
            Timestamp = DateTime.UtcNow,
            WorldSize = 2,
            DdpRank = 0,
            Sharding = new ShardingMetadata
            {
                Strategy = "fsdp",
                ShardCount = 2,
                Precision = "fp16"
            },
            Shards = new List<ShardMetadata>
            {
                new ShardMetadata
                {
                    Rank = 0,
                    FilePath = "shard_0.bin",
                    FileSize = 1024,
                    Tensors = new List<TensorMetadata>(),
                    Checksum = "abc123"
                },
                new ShardMetadata
                {
                    Rank = 1,
                    FilePath = "shard_1.bin",
                    FileSize = 1024,
                    Tensors = new List<TensorMetadata>(),
                    Checksum = "def456"
                }
            }
        };
    }
}
```

### 3. State Dict Tests (`StateDictTests.cs`)
```csharp
[TestFixture]
public class StateDictTests
{
    [Test]
    public void GetTensor_WithExistingKey_ReturnsTensor()
    {
        // Arrange
        var stateDict = new StateDict();
        var tensor = new Tensor { Shape = new[] { 10L, 20L } };
        stateDict["weight"] = tensor;

        // Act
        var result = stateDict.GetTensor("weight");

        // Assert
        Assert.That(result, Is.EqualTo(tensor));
    }

    [Test]
    public void GetTensor_WithNonExistingKey_ThrowsException()
    {
        // Arrange
        var stateDict = new StateDict();

        // Act & Assert
        Assert.Throws<KeyNotFoundException>(() => stateDict.GetTensor("nonexistent"));
    }

    [Test]
    public void GetTensorOrNull_WithExistingKey_ReturnsTensor()
    {
        // Arrange
        var stateDict = new StateDict();
        var tensor = new Tensor { Shape = new[] { 10L, 20L } };
        stateDict["weight"] = tensor;

        // Act
        var result = stateDict.GetTensorOrNull("weight");

        // Assert
        Assert.That(result, Is.EqualTo(tensor));
    }

    [Test]
    public void GetTensorOrNull_WithNonExistingKey_ReturnsNull()
    {
        // Arrange
        var stateDict = new StateDict();

        // Act
        var result = stateDict.GetTensorOrNull("nonexistent");

        // Assert
        Assert.That(result, Is.Null);
    }

    [Test]
    public void StateUtils_KeysMatch_WithSameKeys_ReturnsTrue()
    {
        // Arrange
        var state1 = new StateDict
        {
            ["weight"] = new Tensor(),
            ["bias"] = new Tensor()
        };
        var state2 = new StateDict
        {
            ["weight"] = new Tensor(),
            ["bias"] = new Tensor()
        };

        // Act
        var result = StateUtils.KeysMatch(state1, state2);

        // Assert
        Assert.That(result, Is.True);
    }

    [Test]
    public void StateUtils_ShapesMatch_WithMatchingShapes_ReturnsTrue()
    {
        // Arrange
        var state1 = new StateDict
        {
            ["weight"] = new Tensor { Shape = new[] { 10L, 20L } }
        };
        var state2 = new StateDict
        {
            ["weight"] = new Tensor { Shape = new[] { 10L, 20L } }
        };

        // Act
        var result = StateUtils.ShapesMatch(state1, state2);

        // Assert
        Assert.That(result, Is.True);
    }
}
```

### 4. Validation Tests (`ValidationTests.cs`)
```csharp
[TestFixture]
public class ChecksumIntegrityCheckerTests
{
    private ChecksumIntegrityChecker _checker;

    [SetUp]
    public void SetUp()
    {
        _checker = new ChecksumIntegrityChecker();
    }

    [Test]
    public async Task CheckIntegrityAsync_WithMatchingChecksum_ReturnsValid()
    {
        // Arrange
        var data = Encoding.UTF8.GetBytes("test data");
        var checksum = ComputeChecksum(data);
        var shardMeta = new ShardMetadata
        {
            Rank = 0,
            Checksum = checksum
        };

        // Act
        var result = await _checker.CheckIntegrityAsync(data, shardMeta);

        // Assert
        Assert.That(result.IsValid, Is.True);
        Assert.That(result.Errors, Is.Empty);
    }

    [Test]
    public async Task CheckIntegrityAsync_WithMismatchingChecksum_ReturnsInvalid()
    {
        // Arrange
        var data = Encoding.UTF8.GetBytes("test data");
        var shardMeta = new ShardMetadata
        {
            Rank = 0,
            Checksum = "invalid_checksum"
        };

        // Act
        var result = await _checker.CheckIntegrityAsync(data, shardMeta);

        // Assert
        Assert.That(result.IsValid, Is.False);
        Assert.That(result.Errors, Has.Some.Contains("checksum mismatch"));
    }

    private string ComputeChecksum(byte[] data)
    {
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(data);
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
    }
}
```

### 5. Resharding Tests (`ReshardingTests.cs`)
```csharp
[TestFixture]
public class SimpleReshardingStrategyTests
{
    private SimpleReshardingStrategy _strategy;

    [SetUp]
    public void SetUp()
    {
        _strategy = new SimpleReshardingStrategy();
    }

    [Test]
    public void CreatePlan_WithDifferentWorldSizes_CreatesValidPlan()
    {
        // Arrange
        var metadata = CreateMetadata(sourceWorldSize: 4);
        var targetWorldSize = 2;

        // Act
        var plan = _strategy.CreatePlan(metadata, targetWorldSize);

        // Assert
        Assert.That(plan.SourceWorldSize, Is.EqualTo(4));
        Assert.That(plan.TargetWorldSize, Is.EqualTo(2));
        Assert.That(plan.TensorRedistributions, Is.Not.Empty);
    }

    [Test]
    public void ExecuteAsync_WithValidPlan_ReturnsReshardedData()
    {
        // Arrange
        var plan = new ReshardingPlan
        {
            SourceWorldSize = 2,
            TargetWorldSize = 4,
            TensorRedistributions = new List<TensorRedistribution>()
        };
        var sourceShards = new List<ShardData>
        {
            new ShardData { Data = Encoding.UTF8.GetBytes("shard0") },
            new ShardData { Data = Encoding.UTF8.GetBytes("shard1") }
        };

        // Act
        var task = _strategy.ExecuteAsync(plan, sourceShards);
        var result = task.GetAwaiter().GetResult();

        // Assert
        Assert.That(result.Success, Is.True);
        Assert.That(result.TargetShards.Count, Is.EqualTo(4));
    }

    private static CheckpointMetadata CreateMetadata(int sourceWorldSize)
    {
        var metadata = new CheckpointMetadata
        {
            Sharding = new ShardingMetadata
            {
                ShardCount = sourceWorldSize
            },
            Shards = new List<ShardMetadata>()
        };

        for (int rank = 0; rank < sourceWorldSize; rank++)
        {
            metadata.Shards.Add(new ShardMetadata
            {
                Rank = rank,
                Tensors = new List<TensorMetadata>
                {
                    new TensorMetadata
                    {
                        Name = $"tensor_{rank}",
                        Shape = new[] { 10L }
                    }
                }
            });
        }

        return metadata;
    }
}
```

### 6. Fault Tolerance Tests (`FaultToleranceTests.cs`)
```csharp
[TestFixture]
public class FaultToleranceHandlerTests
{
    private Mock<ICheckpointStorage> _mockStorage;
    private FaultToleranceHandler _handler;

    [SetUp]
    public void SetUp()
    {
        _mockStorage = new Mock<ICheckpointStorage>();
        _handler = new FaultToleranceHandler(_mockStorage.Object);
    }

    [Test]
    public async Task ExecuteWithRetryAsync_WithSuccess_ReturnsResult()
    {
        // Arrange
        var expected = "success";
        var operation = Task.FromResult(expected);

        // Act
        var result = await _handler.ExecuteWithRetryAsync(() => operation);

        // Assert
        Assert.That(result, Is.EqualTo(expected));
    }

    [Test]
    public async Task ExecuteWithRetryAsync_WithRetryableException_Retries()
    {
        // Arrange
        var callCount = 0;
        var operation = Task<string>.Run(async () =>
        {
            callCount++;
            if (callCount < 3)
            {
                throw new IOException("Simulated failure");
            }
            return "success";
        });

        // Act
        var result = await _handler.ExecuteWithRetryAsync(() => operation);

        // Assert
        Assert.That(result, Is.EqualTo("success"));
        Assert.That(callCount, Is.EqualTo(3));
    }

    [Test]
    public async Task ExecuteWithTimeoutAsync_WithTimeout_ThrowsTimeoutException()
    {
        // Arrange
        var operation = Task.Delay(TimeSpan.FromMinutes(10));

        // Act & Assert
        Assert.ThrowsAsync<TimeoutException>(
            () => _handler.ExecuteWithTimeoutAsync(() => operation, TimeSpan.FromMilliseconds(100)));
    }
}
```

### 7. Integration Tests (`CheckpointIntegrationTests.cs`)
```csharp
[TestFixture]
public class DistributedCheckpointIntegrationTests
{
    private string _testDirectory;
    private ICheckpointStorage _storage;
    private Mock<IDistributedCoordinator> _mockCoordinator;

    [SetUp]
    public void SetUp()
    {
        _testDirectory = Path.Combine(Path.GetTempPath(), $"checkpoint_integration_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_testDirectory);

        _storage = new LocalFileSystemStorage(_testDirectory);
        _mockCoordinator = new Mock<IDistributedCoordinator>();
        _mockCoordinator.Setup(c => c.Rank).Returns(0);
        _mockCoordinator.Setup(c => c.WorldSize).Returns(1);
        _mockCoordinator.Setup(c => c.BarrierAsync(It.IsAny<CancellationToken>()))
            .Returns(Task.CompletedTask);
    }

    [TearDown]
    public void TearDown()
    {
        if (Directory.Exists(_testDirectory))
        {
            Directory.Delete(_testDirectory, recursive: true);
        }
    }

    [Test]
    public async Task SaveAndLoad_Roundtrip_Success()
    {
        // Arrange
        var model = new MockModel();
        var optimizer = new MockOptimizer();
        var checkpoint = new DistributedCheckpoint(_mockCoordinator.Object, _storage);

        // Act
        var checkpointPath = await checkpoint.SaveAsync(model, optimizer);
        var loadResult = await checkpoint.LoadAsync(model, optimizer, new LoadOptions
        {
            CheckpointPrefix = checkpointPath
        });

        // Assert
        Assert.That(loadResult.Metadata, Is.Not.Null);
    }

    private class MockModel : IStateful
    {
        public StateDict GetStateDict()
        {
            return new StateDict
            {
                ["weight"] = new Tensor { Shape = new[] { 10L, 20L } }
            };
        }

        public void LoadStateDict(StateDict state)
        {
            // Load state
        }
    }

    private class MockOptimizer : IStateful
    {
        public StateDict GetStateDict()
        {
            return new StateDict
            {
                ["state"] = new Tensor { Shape = new[] { 10L, 20L } }
            };
        }

        public void LoadStateDict(StateDict state)
        {
            // Load state
        }
    }
}
```

## Test Coverage Goals

- **Unit Tests**: 80%+ code coverage
- **Integration Tests**: Core workflows
- **Edge Cases**: Null checks, empty data, large files, timeouts
- **Error Handling**: Exceptions, retries, rollbacks
- **Thread Safety**: Concurrent access scenarios

## Testing Framework
- NUnit or xUnit
- Moq for mocking
- FluentAssertions for readable assertions

## Success Criteria
- All tests pass consistently
- High code coverage (>80%)
- Tests are fast and isolated
- Edge cases are covered
- Error scenarios are tested
