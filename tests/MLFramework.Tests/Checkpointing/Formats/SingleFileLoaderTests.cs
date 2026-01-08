namespace MLFramework.Tests.Checkpointing.Formats;

using MachineLearning.Checkpointing;

/// <summary>
/// Tests for the SingleFileLoader
/// </summary>
public class SingleFileLoaderTests
{
    private readonly SingleFileCheckpointFormat _format;
    private readonly MemoryCheckpointStorage _storage;
    private readonly SingleFileLoader _loader;

    public SingleFileLoaderTests()
    {
        _format = new SingleFileCheckpointFormat();
        _storage = new MemoryCheckpointStorage();
        _loader = new SingleFileLoader(_format, _storage);
    }

    [Fact]
    public void Constructor_NullFormat_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new SingleFileLoader(null, _storage));
    }

    [Fact]
    public void Constructor_NullStorage_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new SingleFileLoader(_format, null));
    }

    [Fact]
    public async Task LoadAsync_ExistingCheckpoint_ReturnsShardsAndMetadata()
    {
        // Arrange
        var checkpointPath = "test_checkpoint.checkpoint";
        var shards = CreateTestShards(2);
        var metadata = CreateTestMetadata(2);
        var serializedData = await _format.SerializeAsync(shards, metadata);
        await _storage.WriteAsync(checkpointPath, serializedData);

        // Act
        var (loadedShards, loadedMetadata) = await _loader.LoadAsync(checkpointPath);

        // Assert
        Assert.NotNull(loadedShards);
        Assert.Equal(2, loadedShards.Count);
        Assert.NotNull(loadedMetadata);
        Assert.Equal(metadata.Version, loadedMetadata.Version);
    }

    [Fact]
    public async Task LoadAsync_NonExistentCheckpoint_ThrowsFileNotFoundException()
    {
        // Arrange
        var checkpointPath = "non_existent.checkpoint";

        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            _loader.LoadAsync(checkpointPath));
    }

    [Fact]
    public async Task LoadAsync_EmptyPath_ThrowsArgumentException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _loader.LoadAsync(""));
    }

    [Fact]
    public async Task LoadByPrefixAsync_ValidPrefix_LoadsCheckpoint()
    {
        // Arrange
        var checkpointPrefix = "prefix_checkpoint";
        var shards = CreateTestShards(2);
        var metadata = CreateTestMetadata(2);
        var serializedData = await _format.SerializeAsync(shards, metadata);
        var checkpointPath = $"{checkpointPrefix}.checkpoint";
        await _storage.WriteAsync(checkpointPath, serializedData);

        // Act
        var (loadedShards, _) = await _loader.LoadByPrefixAsync(checkpointPrefix);

        // Assert
        Assert.NotNull(loadedShards);
        Assert.Equal(2, loadedShards.Count);
    }

    [Fact]
    public async Task LoadByPrefixAsync_NonExistentPrefix_ThrowsFileNotFoundException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            _loader.LoadByPrefixAsync("non_existent"));
    }

    [Fact]
    public async Task LoadMetadataAsync_ValidCheckpoint_ReturnsMetadata()
    {
        // Arrange
        var checkpointPath = "metadata_checkpoint.checkpoint";
        var shards = CreateTestShards(1);
        var metadata = CreateTestMetadata(1);
        metadata.Training = new TrainingMetadata
        {
            Epoch = 10,
            Step = 1000,
            LearningRate = 0.001f,
            OptimizerType = "Adam"
        };
        var serializedData = await _format.SerializeAsync(shards, metadata);
        await _storage.WriteAsync(checkpointPath, serializedData);

        // Act
        var loadedMetadata = await _loader.LoadMetadataAsync(checkpointPath);

        // Assert
        Assert.NotNull(loadedMetadata);
        Assert.Equal(metadata.Version, loadedMetadata.Version);
        Assert.Equal(metadata.WorldSize, loadedMetadata.WorldSize);
        Assert.NotNull(loadedMetadata.Training);
        Assert.Equal(10, loadedMetadata.Training.Epoch);
        Assert.Equal(1000, loadedMetadata.Training.Step);
    }

    [Fact]
    public async Task LoadMetadataAsync_NonExistentCheckpoint_ThrowsFileNotFoundException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            _loader.LoadMetadataAsync("non_existent.checkpoint"));
    }

    [Fact]
    public async Task LoadMetadataAsync_InvalidMagicNumber_ThrowsInvalidDataException()
    {
        // Arrange
        var checkpointPath = "invalid_checkpoint.checkpoint";
        var invalidData = System.Text.Encoding.UTF8.GetBytes("INVALID_MAGIC_NUMBER_DATA");
        await _storage.WriteAsync(checkpointPath, invalidData);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidDataException>(() =>
            _loader.LoadMetadataAsync(checkpointPath));
    }

    [Fact]
    public async Task ExistsAsync_ExistingCheckpoint_ReturnsTrue()
    {
        // Arrange
        var checkpointPath = "existing_checkpoint.checkpoint";
        var shards = CreateTestShards(1);
        var metadata = CreateTestMetadata(1);
        var serializedData = await _format.SerializeAsync(shards, metadata);
        await _storage.WriteAsync(checkpointPath, serializedData);

        // Act
        var exists = await _loader.ExistsAsync(checkpointPath);

        // Assert
        Assert.True(exists);
    }

    [Fact]
    public async Task ExistsAsync_NonExistentCheckpoint_ReturnsFalse()
    {
        // Act
        var exists = await _loader.ExistsAsync("non_existent.checkpoint");

        // Assert
        Assert.False(exists);
    }

    [Fact]
    public async Task ExistsAsync_EmptyPath_ThrowsArgumentException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _loader.ExistsAsync(""));
    }

    [Fact]
    public async Task GetCheckpointMetadataAsync_ExistingCheckpoint_ReturnsMetadata()
    {
        // Arrange
        var checkpointPath = "meta_checkpoint.checkpoint";
        var shards = CreateTestShards(1);
        var metadata = CreateTestMetadata(1);
        var serializedData = await _format.SerializeAsync(shards, metadata);
        await _storage.WriteAsync(checkpointPath, serializedData);

        // Act
        var storageMetadata = await _loader.GetCheckpointMetadataAsync(checkpointPath);

        // Assert
        Assert.NotNull(storageMetadata);
        Assert.Equal(serializedData.Length, storageMetadata.Size);
    }

    [Fact]
    public async Task GetCheckpointMetadataAsync_NonExistentCheckpoint_ThrowsFileNotFoundException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            _loader.GetCheckpointMetadataAsync("non_existent.checkpoint"));
    }

    [Fact]
    public async Task LoadLoadMetadataRoundtrip_PreservesData()
    {
        // Arrange
        var checkpointPath = "roundtrip_checkpoint.checkpoint";
        var originalShards = CreateTestShards(3);
        var originalMetadata = CreateTestMetadata(3);
        var serializedData = await _format.SerializeAsync(originalShards, originalMetadata);
        await _storage.WriteAsync(checkpointPath, serializedData);

        // Act
        var loadedShards = await _loader.LoadAsync(checkpointPath);
        var loadedMetadata = await _loader.LoadMetadataAsync(checkpointPath);

        // Assert
        Assert.Equal(originalShards.Count, loadedShards.Item1.Count);
        Assert.Equal(originalMetadata.Version, loadedMetadata.Version);
        Assert.Equal(originalMetadata.WorldSize, loadedMetadata.WorldSize);
    }

    [Fact]
    public async Task LoadAsync_WithCancellation_ThrowsOperationCanceledException()
    {
        // Arrange
        var checkpointPath = "cancellable_checkpoint.checkpoint";
        var shards = CreateTestShards(1);
        var metadata = CreateTestMetadata(1);
        var serializedData = await _format.SerializeAsync(shards, metadata);
        await _storage.WriteAsync(checkpointPath, serializedData);
        var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            _loader.LoadAsync(checkpointPath, cts.Token));
    }

    private List<ShardData> CreateTestShards(int count)
    {
        var shards = new List<ShardData>();
        for (int i = 0; i < count; i++)
        {
            shards.Add(new ShardData
            {
                Rank = i,
                Data = System.Text.Encoding.UTF8.GetBytes($"Test shard {i} data"),
                TensorInfo = new List<TensorMetadata>
                {
                    new TensorMetadata
                    {
                        Name = $"tensor_{i}_0",
                        DataType = TensorDataType.Float32,
                        Shape = new long[] { 100, 200 },
                        Size = 100 * 200 * 4
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
