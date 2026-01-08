namespace MLFramework.Tests.Checkpointing.Formats;

using MachineLearning.Checkpointing;
using System.Text;

/// <summary>
/// Tests for the multi-shard saver and loader
/// </summary>
public class MultiShardSaverLoaderTests
{
    private readonly MultiShardCheckpointFormat _format;
    private readonly MemoryCheckpointStorage _storage;
    private readonly MultiShardSaver _saver;
    private readonly MultiShardLoader _loader;

    public MultiShardSaverLoaderTests()
    {
        _format = new MultiShardCheckpointFormat();
        _storage = new MemoryCheckpointStorage();
        _saver = new MultiShardSaver(_format, _storage);
        _loader = new MultiShardLoader(_format, _storage);
    }

    [Fact]
    public void MultiShardSaver_Constructor_NullFormat_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new MultiShardSaver(null, _storage));
    }

    [Fact]
    public void MultiShardSaver_Constructor_NullStorage_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new MultiShardSaver(_format, null));
    }

    [Fact]
    public void MultiShardLoader_Constructor_NullFormat_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new MultiShardLoader(null, _storage));
    }

    [Fact]
    public void MultiShardLoader_Constructor_NullStorage_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new MultiShardLoader(_format, null));
    }

    [Fact]
    public async Task SaveShardAsync_ValidInput_SavesShard()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_test";
        var rank = 0;
        var shard = CreateTestShard(rank);
        var shardMeta = CreateTestShardMetadata(rank);

        // Act
        var resultPath = await _saver.SaveShardAsync(checkpointPrefix, rank, shard, shardMeta);

        // Assert
        Assert.Equal("checkpoint_test_shard_0.shard", resultPath);
        Assert.True(await _storage.ExistsAsync(resultPath));
    }

    [Fact]
    public async Task SaveShardAsync_EmptyPrefix_ThrowsArgumentException()
    {
        // Arrange
        var shard = CreateTestShard(0);
        var shardMeta = CreateTestShardMetadata(0);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _saver.SaveShardAsync("", 0, shard, shardMeta));
    }

    [Fact]
    public async Task SaveShardAsync_NullShard_ThrowsArgumentNullException()
    {
        // Arrange
        var shardMeta = CreateTestShardMetadata(0);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _saver.SaveShardAsync("checkpoint", 0, null, shardMeta));
    }

    [Fact]
    public async Task SaveShardAsync_NullMetadata_ThrowsArgumentNullException()
    {
        // Arrange
        var shard = CreateTestShard(0);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _saver.SaveShardAsync("checkpoint", 0, shard, null));
    }

    [Fact]
    public async Task SaveShardAsync_MultipleRanks_SavesMultipleShards()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_multi";
        var shardCount = 4;

        // Act
        for (int i = 0; i < shardCount; i++)
        {
            var shard = CreateTestShard(i);
            var shardMeta = CreateTestShardMetadata(i);
            await _saver.SaveShardAsync(checkpointPrefix, i, shard, shardMeta);
        }

        // Assert
        for (int i = 0; i < shardCount; i++)
        {
            var path = $"{checkpointPrefix}_shard_{i}.shard";
            Assert.True(await _storage.ExistsAsync(path));
        }
    }

    [Fact]
    public async Task SaveMetadataAsync_ValidInput_SavesMetadata()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_meta";
        var metadata = CreateTestMetadata(2);

        // Act
        var resultPath = await _saver.SaveMetadataAsync(checkpointPrefix, metadata);

        // Assert
        Assert.Equal("checkpoint_meta.metadata.json", resultPath);
        Assert.True(await _storage.ExistsAsync(resultPath));
    }

    [Fact]
    public async Task SaveMetadataAsync_EmptyPrefix_ThrowsArgumentException()
    {
        // Arrange
        var metadata = CreateTestMetadata(2);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _saver.SaveMetadataAsync("", metadata));
    }

    [Fact]
    public async Task SaveMetadataAsync_NullMetadata_ThrowsArgumentNullException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _saver.SaveMetadataAsync("checkpoint", null));
    }

    [Fact]
    public async Task LoadShardAsync_ValidInput_LoadsShard()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_load";
        var rank = 0;
        var shard = CreateTestShard(rank);
        var shardMeta = CreateTestShardMetadata(rank);

        // Save the shard first
        await _saver.SaveShardAsync(checkpointPrefix, rank, shard, shardMeta);

        // Act
        var loadedShard = await _loader.LoadShardAsync(checkpointPrefix, rank, shardMeta);

        // Assert
        Assert.NotNull(loadedShard);
        Assert.Equal(rank, loadedShard.Rank);
        Assert.True(shard.Data.SequenceEqual(loadedShard.Data));
    }

    [Fact]
    public async Task LoadShardAsync_EmptyPrefix_ThrowsArgumentException()
    {
        // Arrange
        var shardMeta = CreateTestShardMetadata(0);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _loader.LoadShardAsync("", 0, shardMeta));
    }

    [Fact]
    public async Task LoadShardAsync_NullMetadata_ThrowsArgumentNullException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _loader.LoadShardAsync("checkpoint", 0, null));
    }

    [Fact]
    public async Task LoadShardAsync_NonExistentFile_ThrowsFileNotFoundException()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_nonexistent";
        var shardMeta = CreateTestShardMetadata(0);

        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            _loader.LoadShardAsync(checkpointPrefix, 0, shardMeta));
    }

    [Fact]
    public async Task LoadMetadataAsync_ValidInput_LoadsMetadata()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_meta_load";
        var metadata = CreateTestMetadata(3);

        // Save the metadata first
        await _saver.SaveMetadataAsync(checkpointPrefix, metadata);

        // Act
        var loadedMetadata = await _loader.LoadMetadataAsync(checkpointPrefix);

        // Assert
        Assert.NotNull(loadedMetadata);
        Assert.Equal(metadata.Version, loadedMetadata.Version);
        Assert.Equal(metadata.WorldSize, loadedMetadata.WorldSize);
        Assert.NotNull(loadedMetadata.Shards);
    }

    [Fact]
    public async Task LoadMetadataAsync_EmptyPrefix_ThrowsArgumentException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _loader.LoadMetadataAsync(""));
    }

    [Fact]
    public async Task LoadMetadataAsync_NonExistentFile_ThrowsFileNotFoundException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            _loader.LoadMetadataAsync("checkpoint_nonexistent"));
    }

    [Fact]
    public async Task LoadAllShardsAsync_ValidInput_LoadsAllShards()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_all";
        var shardCount = 3;
        var metadata = CreateTestMetadata(shardCount);

        // Save all shards
        for (int i = 0; i < shardCount; i++)
        {
            var shard = CreateTestShard(i);
            var shardMeta = CreateTestShardMetadata(i);
            await _saver.SaveShardAsync(checkpointPrefix, i, shard, shardMeta);
        }

        // Act
        var loadedShards = await _loader.LoadAllShardsAsync(checkpointPrefix, metadata);

        // Assert
        Assert.NotNull(loadedShards);
        Assert.Equal(shardCount, loadedShards.Count);
        for (int i = 0; i < shardCount; i++)
        {
            Assert.Equal(i, loadedShards[i].Rank);
        }
    }

    [Fact]
    public async Task LoadAllShardsAsync_EmptyPrefix_ThrowsArgumentException()
    {
        // Arrange
        var metadata = CreateTestMetadata(2);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _loader.LoadAllShardsAsync("", metadata));
    }

    [Fact]
    public async Task LoadAllShardsAsync_NullMetadata_ThrowsArgumentNullException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _loader.LoadAllShardsAsync("checkpoint", null));
    }

    [Fact]
    public async Task LoadAllShardsAsync_MetadataWithNoShards_ReturnsEmptyList()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_empty";
        var metadata = CreateTestMetadata(0);

        // Act
        var loadedShards = await _loader.LoadAllShardsAsync(checkpointPrefix, metadata);

        // Assert
        Assert.NotNull(loadedShards);
        Assert.Empty(loadedShards);
    }

    [Fact]
    public async Task SaveLoad_RoundTrip_PreservesData()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_roundtrip";
        var rank = 0;
        var shard = CreateTestShard(rank);
        var shardMeta = CreateTestShardMetadata(rank);

        // Act
        await _saver.SaveShardAsync(checkpointPrefix, rank, shard, shardMeta);
        var loadedShard = await _loader.LoadShardAsync(checkpointPrefix, rank, shardMeta);

        // Assert
        Assert.True(shard.Data.SequenceEqual(loadedShard.Data));
        Assert.Equal(shard.TensorInfo.Count, loadedShard.TensorInfo.Count);
    }

    [Fact]
    public async Task SaveLoad_RoundTrip_PreservesMetadata()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_meta_roundtrip";
        var metadata = CreateTestMetadata(2);

        // Act
        await _saver.SaveMetadataAsync(checkpointPrefix, metadata);
        var loadedMetadata = await _loader.LoadMetadataAsync(checkpointPrefix);

        // Assert
        Assert.Equal(metadata.Version, loadedMetadata.Version);
        Assert.Equal(metadata.WorldSize, loadedMetadata.WorldSize);
        Assert.Equal(metadata.DdpRank, loadedMetadata.DdpRank);
    }

    [Fact]
    public async Task SaveLoad_MultipleShards_RoundTrip_PreservesAllData()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_multi_roundtrip";
        var shardCount = 4;
        var originalShards = new List<ShardData>();
        var metadata = CreateTestMetadata(shardCount);

        // Save all shards
        for (int i = 0; i < shardCount; i++)
        {
            var shard = CreateTestShard(i);
            var shardMeta = CreateTestShardMetadata(i);
            await _saver.SaveShardAsync(checkpointPrefix, i, shard, shardMeta);
            originalShards.Add(shard);
        }

        // Act
        var loadedShards = await _loader.LoadAllShardsAsync(checkpointPrefix, metadata);

        // Assert
        Assert.Equal(shardCount, loadedShards.Count);
        for (int i = 0; i < shardCount; i++)
        {
            Assert.True(originalShards[i].Data.SequenceEqual(loadedShards[i].Data));
        }
    }

    private ShardData CreateTestShard(int rank)
    {
        return new ShardData
        {
            Rank = rank,
            Data = Encoding.UTF8.GetBytes($"Test shard {rank} data"),
            TensorInfo = new List<TensorMetadata>
            {
                new TensorMetadata
                {
                    Name = $"tensor_{rank}_0",
                    DataType = TensorDataType.Float32,
                    Shape = new long[] { 100, 200 },
                    Size = 100 * 200 * 4
                }
            }
        };
    }

    private ShardMetadata CreateTestShardMetadata(int rank)
    {
        return new ShardMetadata
        {
            Rank = rank,
            FilePath = $"shard_{rank}.shard",
            FileSize = 1024,
            Tensors = new List<TensorMetadata>
            {
                new TensorMetadata
                {
                    Name = $"tensor_{rank}_0",
                    DataType = TensorDataType.Float32,
                    Shape = new long[] { 100, 200 },
                    Size = 100 * 200 * 4
                }
            }
        };
    }

    private CheckpointMetadata CreateTestMetadata(int shardCount)
    {
        var metadata = new CheckpointMetadata
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

        for (int i = 0; i < shardCount; i++)
        {
            metadata.Shards.Add(CreateTestShardMetadata(i));
        }

        return metadata;
    }
}
