namespace MachineLearning.Checkpointing.Tests;

using Xunit;

/// <summary>
/// Tests for metadata serialization and validation
/// </summary>
public class MetadataTests
{
    [Fact]
    public void SerializeAndDeserialize_Roundtrip_Success()
    {
        // Arrange
        var expectedMetadata = CreateTestMetadata();

        // Act
        var json = MetadataSerializer.Serialize(expectedMetadata);
        var actualMetadata = MetadataSerializer.Deserialize(json);

        // Assert
        Assert.Equal(expectedMetadata.Version, actualMetadata.Version);
        Assert.Equal(expectedMetadata.WorldSize, actualMetadata.WorldSize);
    }

    [Fact]
    public void Validate_WithValidMetadata_ReturnsValid()
    {
        // Arrange
        var metadata = CreateTestMetadata();

        // Act
        var result = MetadataValidator.Validate(metadata);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public void Validate_WithMissingVersion_ReturnsInvalid()
    {
        // Arrange
        var metadata = CreateTestMetadata();
        metadata.Version = null;

        // Act
        var result = MetadataValidator.Validate(metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("Version"));
    }

    [Fact]
    public void Validate_WithShardCountMismatch_ReturnsInvalid()
    {
        // Arrange
        var metadata = CreateTestMetadata();
        metadata.Sharding!.ShardCount = 4;
        metadata.Shards = new List<ShardMetadata>(new ShardMetadata[2]);

        // Act
        var result = MetadataValidator.Validate(metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("Shard count"));
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
