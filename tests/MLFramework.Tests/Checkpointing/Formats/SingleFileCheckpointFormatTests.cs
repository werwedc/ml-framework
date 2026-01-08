namespace MLFramework.Tests.Checkpointing.Formats;

using MachineLearning.Checkpointing;
using System.Text;

/// <summary>
/// Tests for the single-file checkpoint format implementation
/// </summary>
public class SingleFileCheckpointFormatTests
{
    private readonly SingleFileCheckpointFormat _format;
    private readonly MemoryCheckpointStorage _storage;

    public SingleFileCheckpointFormatTests()
    {
        _format = new SingleFileCheckpointFormat();
        _storage = new MemoryCheckpointStorage();
    }

    [Fact]
    public void Extension_ReturnsCorrectValue()
    {
        Assert.Equal(".checkpoint", _format.Extension);
    }

    [Fact]
    public async Task SerializeAsync_ValidInput_ReturnsData()
    {
        // Arrange
        var shards = CreateTestShards(2);
        var metadata = CreateTestMetadata(2);

        // Act
        var result = await _format.SerializeAsync(shards, metadata);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.Length > 0);
    }

    [Fact]
    public async Task SerializeAsync_NullShards_ThrowsArgumentNullException()
    {
        // Arrange
        var metadata = CreateTestMetadata(2);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _format.SerializeAsync(null, metadata));
    }

    [Fact]
    public async Task SerializeAsync_NullMetadata_ThrowsArgumentNullException()
    {
        // Arrange
        var shards = CreateTestShards(2);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _format.SerializeAsync(shards, null));
    }

    [Fact]
    public async Task DeserializeAsync_ValidData_ReturnsShardsAndMetadata()
    {
        // Arrange
        var shards = CreateTestShards(2);
        var metadata = CreateTestMetadata(2);
        var serializedData = await _format.SerializeAsync(shards, metadata);

        // Act
        var (resultShards, resultMetadata) = await _format.DeserializeAsync(serializedData);

        // Assert
        Assert.NotNull(resultShards);
        Assert.Equal(2, resultShards.Count);
        Assert.NotNull(resultMetadata);
        Assert.Equal(metadata.Version, resultMetadata.Version);
        Assert.Equal(metadata.WorldSize, resultMetadata.WorldSize);
    }

    [Fact]
    public async Task DeserializeAsync_EmptyData_ThrowsArgumentException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _format.DeserializeAsync(Array.Empty<byte>()));
    }

    [Fact]
    public async Task DeserializeAsync_NullData_ThrowsArgumentException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _format.DeserializeAsync(null));
    }

    [Fact]
    public async Task DeserializeAsync_InvalidMagicNumber_ThrowsInvalidDataException()
    {
        // Arrange
        var invalidData = Encoding.UTF8.GetBytes("INVALID_MAGIC_NUMBER_DATA");

        // Act & Assert
        await Assert.ThrowsAsync<InvalidDataException>(() =>
            _format.DeserializeAsync(invalidData));
    }

    [Fact]
    public async Task SerializeDeserialize_RoundTrip_PreservesMetadata()
    {
        // Arrange
        var shards = CreateTestShards(3);
        var metadata = CreateTestMetadata(3);

        // Act
        var serialized = await _format.SerializeAsync(shards, metadata);
        var (deserializedShards, deserializedMetadata) = await _format.DeserializeAsync(serialized);

        // Assert
        Assert.Equal(metadata.Version, deserializedMetadata.Version);
        Assert.Equal(metadata.WorldSize, deserializedMetadata.WorldSize);
        Assert.NotNull(deserializedMetadata.Shards);
        Assert.Equal(metadata.Shards.Count, deserializedMetadata.Shards.Count);
    }

    [Fact]
    public async Task SerializeDeserialize_RoundTrip_PreservesShardCount()
    {
        // Arrange
        var shards = CreateTestShards(4);
        var metadata = CreateTestMetadata(4);

        // Act
        var serialized = await _format.SerializeAsync(shards, metadata);
        var (deserializedShards, _) = await _format.DeserializeAsync(serialized);

        // Assert
        Assert.Equal(4, deserializedShards.Count);
    }

    [Fact]
    public async Task SerializeDeserialize_RoundTrip_PreservesTensorInfo()
    {
        // Arrange
        var shards = CreateTestShards(2);
        var metadata = CreateTestMetadata(2);

        // Act
        var serialized = await _format.SerializeAsync(shards, metadata);
        var (deserializedShards, _) = await _format.DeserializeAsync(serialized);

        // Assert
        for (int i = 0; i < shards.Count; i++)
        {
            Assert.Equal(shards[i].TensorInfo.Count, deserializedShards[i].TensorInfo.Count);
            for (int j = 0; j < shards[i].TensorInfo.Count; j++)
            {
                Assert.Equal(shards[i].TensorInfo[j].Name, deserializedShards[i].TensorInfo[j].Name);
                Assert.Equal(shards[i].TensorInfo[j].DataType, deserializedShards[i].TensorInfo[j].DataType);
                Assert.True(shards[i].TensorInfo[j].Shape.SequenceEqual(deserializedShards[i].TensorInfo[j].Shape));
            }
        }
    }

    [Fact]
    public async Task SerializeDeserialize_WithTrainingMetadata_PreservesTrainingState()
    {
        // Arrange
        var shards = CreateTestShards(1);
        var metadata = CreateTestMetadata(1);
        metadata.Training = new TrainingMetadata
        {
            Epoch = 10,
            Step = 1000,
            LearningRate = 0.001f,
            OptimizerType = "Adam"
        };

        // Act
        var serialized = await _format.SerializeAsync(shards, metadata);
        var (_, deserializedMetadata) = await _format.DeserializeAsync(serialized);

        // Assert
        Assert.NotNull(deserializedMetadata.Training);
        Assert.Equal(10, deserializedMetadata.Training.Epoch);
        Assert.Equal(1000, deserializedMetadata.Training.Step);
        Assert.Equal(0.001f, deserializedMetadata.Training.LearningRate);
        Assert.Equal("Adam", deserializedMetadata.Training.OptimizerType);
    }

    [Fact]
    public async Task SerializeDeserialize_WithCustomFields_PreservesCustomFields()
    {
        // Arrange
        var shards = CreateTestShards(1);
        var metadata = CreateTestMetadata(1);
        metadata.CustomFields = new Dictionary<string, string>
        {
            { "experiment_name", "test_experiment" },
            { "author", "test_user" }
        };

        // Act
        var serialized = await _format.SerializeAsync(shards, metadata);
        var (_, deserializedMetadata) = await _format.DeserializeAsync(serialized);

        // Assert
        Assert.NotNull(deserializedMetadata.CustomFields);
        Assert.Equal(2, deserializedMetadata.CustomFields.Count);
        Assert.Equal("test_experiment", deserializedMetadata.CustomFields["experiment_name"]);
        Assert.Equal("test_user", deserializedMetadata.CustomFields["author"]);
    }

    private List<ShardData> CreateTestShards(int count)
    {
        var shards = new List<ShardData>();
        for (int i = 0; i < count; i++)
        {
            shards.Add(new ShardData
            {
                Rank = i,
                Data = Encoding.UTF8.GetBytes($"Test shard {i} data"),
                TensorInfo = new List<TensorMetadata>
                {
                    new TensorMetadata
                    {
                        Name = $"tensor_{i}_0",
                        DataType = TensorDataType.Float32,
                        Shape = new long[] { 100, 200 },
                        Size = 100 * 200 * 4
                    },
                    new TensorMetadata
                    {
                        Name = $"tensor_{i}_1",
                        DataType = TensorDataType.Float32,
                        Shape = new long[] { 50, 100 },
                        Size = 50 * 100 * 4
                    }
                }
            });
        }
        return shards;
    }

    private CheckpointMetadata CreateTestMetadata(int shardCount)
    {
        return new CheckpointMetadata
        {
            Version = "1.0.0",
            WorldSize = shardCount,
            DdpRank = 0,
            Timestamp = DateTime.UtcNow,
            Shards = new List<ShardMetadata>(),
            Sharding = new ShardingMetadata
            {
                Strategy = "test_strategy",
                ShardCount = shardCount,
                Precision = "float32"
            }
        };
    }
}

/// <summary>
/// In-memory storage implementation for testing
/// </summary>
public class MemoryCheckpointStorage : ICheckpointStorage
{
    private readonly Dictionary<string, byte[]> _storage = new();

    public Task WriteAsync(string path, byte[] data, CancellationToken cancellationToken = default)
    {
        _storage[path] = data;
        return Task.CompletedTask;
    }

    public Task<byte[]> ReadAsync(string path, CancellationToken cancellationToken = default)
    {
        if (!_storage.ContainsKey(path))
            throw new FileNotFoundException($"File not found: {path}");
        return Task.FromResult(_storage[path]);
    }

    public Task<bool> ExistsAsync(string path, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(_storage.ContainsKey(path));
    }

    public Task DeleteAsync(string path, CancellationToken cancellationToken = default)
    {
        _storage.Remove(path);
        return Task.CompletedTask;
    }

    public Task<StorageMetadata> GetMetadataAsync(string path, CancellationToken cancellationToken = default)
    {
        if (!_storage.ContainsKey(path))
            throw new FileNotFoundException($"File not found: {path}");
        return Task.FromResult(new StorageMetadata
        {
            Size = _storage[path].Length,
            LastModified = DateTime.UtcNow
        });
    }
}
