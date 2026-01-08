namespace MLFramework.Tests.Checkpointing.Formats;

using MachineLearning.Checkpointing;
using System.Text;

/// <summary>
/// Tests for the multi-shard checkpoint format implementation
/// </summary>
public class MultiShardCheckpointFormatTests
{
    private readonly MultiShardCheckpointFormat _format;

    public MultiShardCheckpointFormatTests()
    {
        _format = new MultiShardCheckpointFormat();
    }

    [Fact]
    public void Extension_ReturnsCorrectValue()
    {
        Assert.Equal(".shard", _format.Extension);
    }

    [Fact]
    public async Task SerializeAsync_ThrowsNotSupportedException()
    {
        // Arrange
        var shards = CreateTestShards(2);
        var metadata = CreateTestMetadata(2);

        // Act & Assert
        await Assert.ThrowsAsync<NotSupportedException>(() =>
            _format.SerializeAsync(shards, metadata));
    }

    [Fact]
    public async Task DeserializeAsync_ThrowsNotSupportedException()
    {
        // Arrange
        var data = Encoding.UTF8.GetBytes("test data");

        // Act & Assert
        await Assert.ThrowsAsync<NotSupportedException>(() =>
            _format.DeserializeAsync(data));
    }

    [Fact]
    public async Task SerializeShardAsync_ValidInput_ReturnsData()
    {
        // Arrange
        var shard = CreateTestShard(0);
        var shardMeta = CreateTestShardMetadata(0);

        // Act
        var result = await _format.SerializeShardAsync(shard, shardMeta);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.Length > 0);
    }

    [Fact]
    public async Task SerializeShardAsync_NullShard_ThrowsArgumentNullException()
    {
        // Arrange
        var shardMeta = CreateTestShardMetadata(0);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _format.SerializeShardAsync(null, shardMeta));
    }

    [Fact]
    public async Task SerializeShardAsync_NullMetadata_ThrowsArgumentNullException()
    {
        // Arrange
        var shard = CreateTestShard(0);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _format.SerializeShardAsync(shard, null));
    }

    [Fact]
    public async Task DeserializeShardAsync_ValidData_ReturnsShard()
    {
        // Arrange
        var shard = CreateTestShard(0);
        var shardMeta = CreateTestShardMetadata(0);
        var serializedData = await _format.SerializeShardAsync(shard, shardMeta);

        // Act
        var result = await _format.DeserializeShardAsync(serializedData, shardMeta);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(shard.Rank, result.Rank);
        Assert.True(shard.Data.SequenceEqual(result.Data));
        Assert.Equal(shard.TensorInfo.Count, result.TensorInfo.Count);
    }

    [Fact]
    public async Task DeserializeShardAsync_EmptyData_ThrowsArgumentException()
    {
        // Arrange
        var shardMeta = CreateTestShardMetadata(0);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _format.DeserializeShardAsync(Array.Empty<byte>(), shardMeta));
    }

    [Fact]
    public async Task DeserializeShardAsync_NullData_ThrowsArgumentException()
    {
        // Arrange
        var shardMeta = CreateTestShardMetadata(0);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _format.DeserializeShardAsync(null, shardMeta));
    }

    [Fact]
    public async Task DeserializeShardAsync_NullMetadata_ThrowsArgumentNullException()
    {
        // Arrange
        var shard = CreateTestShard(0);
        var shardMeta = CreateTestShardMetadata(0);
        var serializedData = await _format.SerializeShardAsync(shard, shardMeta);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _format.DeserializeShardAsync(serializedData, null));
    }

    [Fact]
    public async Task DeserializeShardAsync_InvalidMagicNumber_ThrowsInvalidDataException()
    {
        // Arrange
        var invalidData = Encoding.UTF8.GetBytes("INVALID_MAGIC_NUMBER_DATA");
        var shardMeta = CreateTestShardMetadata(0);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidDataException>(() =>
            _format.DeserializeShardAsync(invalidData, shardMeta));
    }

    [Fact]
    public async Task DeserializeShardAsync_RankMismatch_ThrowsInvalidDataException()
    {
        // Arrange
        var shard = CreateTestShard(0);
        var shardMeta = CreateTestShardMetadata(0);
        var serializedData = await _format.SerializeShardAsync(shard, shardMeta);

        // Create metadata with different rank
        var wrongRankMeta = CreateTestShardMetadata(1);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidDataException>(() =>
            _format.DeserializeShardAsync(serializedData, wrongRankMeta));
    }

    [Fact]
    public async Task DeserializeShardAsync_TensorCountMismatch_ThrowsInvalidDataException()
    {
        // Arrange
        var shard = CreateTestShard(0);
        var shardMeta = CreateTestShardMetadata(0);
        var serializedData = await _format.SerializeShardAsync(shard, shardMeta);

        // Create metadata with different tensor count
        var wrongTensorCountMeta = CreateTestShardMetadata(0);
        wrongTensorCountMeta.Tensors.Add(new TensorMetadata
        {
            Name = "extra_tensor",
            DataType = TensorDataType.Float32,
            Shape = new long[] { 10, 10 },
            Size = 400
        });

        // Act & Assert
        await Assert.ThrowsAsync<InvalidDataException>(() =>
            _format.DeserializeShardAsync(serializedData, wrongTensorCountMeta));
    }

    [Fact]
    public async Task SerializeDeserialize_RoundTrip_PreservesData()
    {
        // Arrange
        var shard = CreateTestShard(1);
        var shardMeta = CreateTestShardMetadata(1);

        // Act
        var serialized = await _format.SerializeShardAsync(shard, shardMeta);
        var deserialized = await _format.DeserializeShardAsync(serialized, shardMeta);

        // Assert
        Assert.True(shard.Data.SequenceEqual(deserialized.Data));
    }

    [Fact]
    public async Task SerializeDeserialize_RoundTrip_PreservesTensorInfo()
    {
        // Arrange
        var shard = CreateTestShard(2);
        var shardMeta = CreateTestShardMetadata(2);

        // Act
        var serialized = await _format.SerializeShardAsync(shard, shardMeta);
        var deserialized = await _format.DeserializeShardAsync(serialized, shardMeta);

        // Assert
        Assert.Equal(shard.TensorInfo.Count, deserialized.TensorInfo.Count);
        for (int i = 0; i < shard.TensorInfo.Count; i++)
        {
            Assert.Equal(shard.TensorInfo[i].Name, deserialized.TensorInfo[i].Name);
            Assert.Equal(shard.TensorInfo[i].DataType, deserialized.TensorInfo[i].DataType);
            Assert.True(shard.TensorInfo[i].Shape.SequenceEqual(deserialized.TensorInfo[i].Shape));
            Assert.Equal(shard.TensorInfo[i].Size, deserialized.TensorInfo[i].Size);
        }
    }

    [Fact]
    public async Task SerializeDeserialize_RoundTrip_PreservesRank()
    {
        // Arrange
        var shard = CreateTestShard(5);
        var shardMeta = CreateTestShardMetadata(5);

        // Act
        var serialized = await _format.SerializeShardAsync(shard, shardMeta);
        var deserialized = await _format.DeserializeShardAsync(serialized, shardMeta);

        // Assert
        Assert.Equal(5, deserialized.Rank);
    }

    [Fact]
    public async Task SerializeShardAsync_WithChecksum_WritesChecksum()
    {
        // Arrange
        var shard = CreateTestShard(0);
        var shardMeta = CreateTestShardMetadata(0);
        shardMeta.Checksum = "test_checksum_123";

        // Act
        var serialized = await _format.SerializeShardAsync(shard, shardMeta);

        // Assert
        Assert.NotNull(serialized);
        Assert.Contains(Encoding.UTF8.GetBytes("test_checksum_123"), serialized);
    }

    [Fact]
    public async Task DeserializeShardAsync_ChecksumMismatch_ThrowsInvalidDataException()
    {
        // Arrange
        var shard = CreateTestShard(0);
        var shardMeta = CreateTestShardMetadata(0);
        shardMeta.Checksum = "expected_checksum";
        var serialized = await _format.SerializeShardAsync(shard, shardMeta);

        // Create metadata with different checksum
        var wrongChecksumMeta = CreateTestShardMetadata(0);
        wrongChecksumMeta.Checksum = "different_checksum";

        // Act & Assert
        await Assert.ThrowsAsync<InvalidDataException>(() =>
            _format.DeserializeShardAsync(serialized, wrongChecksumMeta));
    }

    [Fact]
    public async Task DeserializeShardAsync_MatchingChecksum_Passes()
    {
        // Arrange
        var shard = CreateTestShard(0);
        var shardMeta = CreateTestShardMetadata(0);
        shardMeta.Checksum = "test_checksum";
        var serialized = await _format.SerializeShardAsync(shard, shardMeta);

        // Use same checksum for deserialization
        var matchingMeta = CreateTestShardMetadata(0);
        matchingMeta.Checksum = "test_checksum";

        // Act & Assert - should not throw
        var result = await _format.DeserializeShardAsync(serialized, matchingMeta);
        Assert.NotNull(result);
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
                },
                new TensorMetadata
                {
                    Name = $"tensor_{rank}_1",
                    DataType = TensorDataType.Float32,
                    Shape = new long[] { 50, 100 },
                    Size = 50 * 100 * 4
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
                },
                new TensorMetadata
                {
                    Name = $"tensor_{rank}_1",
                    DataType = TensorDataType.Float32,
                    Shape = new long[] { 50, 100 },
                    Size = 50 * 100 * 4
                }
            }
        };
    }
}
