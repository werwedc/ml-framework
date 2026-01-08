namespace MLFramework.Tests.Checkpointing.Formats;

using MachineLearning.Checkpointing;

/// <summary>
/// Tests for the SingleFileSaver
/// </summary>
public class SingleFileSaverTests
{
    private readonly SingleFileCheckpointFormat _format;
    private readonly MemoryCheckpointStorage _storage;
    private readonly SingleFileSaver _saver;

    public SingleFileSaverTests()
    {
        _format = new SingleFileCheckpointFormat();
        _storage = new MemoryCheckpointStorage();
        _saver = new SingleFileSaver(_format, _storage);
    }

    [Fact]
    public void Constructor_NullFormat_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new SingleFileSaver(null, _storage));
    }

    [Fact]
    public void Constructor_NullStorage_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new SingleFileSaver(_format, null));
    }

    [Fact]
    public async Task SaveAsync_ValidInput_SavesCheckpoint()
    {
        // Arrange
        var checkpointPrefix = "test_checkpoint";
        var shards = CreateTestShards(2);
        var metadata = CreateTestMetadata(2);

        // Act
        var savedPath = await _saver.SaveAsync(checkpointPrefix, shards, metadata);

        // Assert
        Assert.Equal("test_checkpoint.checkpoint", savedPath);
        Assert.True(await _storage.ExistsAsync(savedPath));

        var savedData = await _storage.ReadAsync(savedPath);
        Assert.True(savedData.Length > 0);
    }

    [Fact]
    public async Task SaveAsync_EmptyPrefix_ThrowsArgumentException()
    {
        // Arrange
        var shards = CreateTestShards(2);
        var metadata = CreateTestMetadata(2);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _saver.SaveAsync("", shards, metadata));
    }

    [Fact]
    public async Task SaveAsync_NullShards_ThrowsArgumentNullException()
    {
        // Arrange
        var metadata = CreateTestMetadata(2);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _saver.SaveAsync("test", null, metadata));
    }

    [Fact]
    public async Task SaveAsync_NullMetadata_ThrowsArgumentNullException()
    {
        // Arrange
        var shards = CreateTestShards(2);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _saver.SaveAsync("test", shards, null));
    }

    [Fact]
    public async Task SaveToPathAsync_ValidInput_SavesToCustomPath()
    {
        // Arrange
        var customPath = "custom/path/checkpoint.checkpoint";
        var shards = CreateTestShards(2);
        var metadata = CreateTestMetadata(2);

        // Act
        var savedPath = await _saver.SaveToPathAsync(customPath, shards, metadata);

        // Assert
        Assert.Equal(customPath, savedPath);
        Assert.True(await _storage.ExistsAsync(savedPath));
    }

    [Fact]
    public async Task SaveToPathAsync_EmptyPath_ThrowsArgumentException()
    {
        // Arrange
        var shards = CreateTestShards(2);
        var metadata = CreateTestMetadata(2);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _saver.SaveToPathAsync("", shards, metadata));
    }

    [Fact]
    public async Task SaveAsync_MultipleShards_WritesAllShards()
    {
        // Arrange
        var checkpointPrefix = "multi_shard_checkpoint";
        var shards = CreateTestShards(5);
        var metadata = CreateTestMetadata(5);

        // Act
        var savedPath = await _saver.SaveAsync(checkpointPrefix, shards, metadata);

        // Assert
        var savedData = await _storage.ReadAsync(savedPath);

        // Deserialize to verify all shards were saved
        var format = new SingleFileCheckpointFormat();
        var (deserializedShards, _) = await format.DeserializeAsync(savedData);
        Assert.Equal(5, deserializedShards.Count);
    }

    [Fact]
    public async Task SaveAsync_WithTrainingMetadata_SavesTrainingState()
    {
        // Arrange
        var checkpointPrefix = "training_checkpoint";
        var shards = CreateTestShards(1);
        var metadata = CreateTestMetadata(1);
        metadata.Training = new TrainingMetadata
        {
            Epoch = 5,
            Step = 500,
            LearningRate = 0.0001f,
            OptimizerType = "SGD"
        };

        // Act
        var savedPath = await _saver.SaveAsync(checkpointPrefix, shards, metadata);

        // Assert
        var savedData = await _storage.ReadAsync(savedPath);
        var format = new SingleFileCheckpointFormat();
        var (_, deserializedMetadata) = await format.DeserializeAsync(savedData);

        Assert.NotNull(deserializedMetadata.Training);
        Assert.Equal(5, deserializedMetadata.Training.Epoch);
        Assert.Equal(500, deserializedMetadata.Training.Step);
        Assert.Equal(0.0001f, deserializedMetadata.Training.LearningRate);
        Assert.Equal("SGD", deserializedMetadata.Training.OptimizerType);
    }

    [Fact]
    public async Task SaveAsync_WithCancellation_ThrowsOperationCanceledException()
    {
        // Arrange
        var checkpointPrefix = "cancellable_checkpoint";
        var shards = CreateTestShards(1);
        var metadata = CreateTestMetadata(1);
        var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act & Assert
        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            _saver.SaveAsync(checkpointPrefix, shards, metadata, cts.Token));
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
