using System.Text.Json;
using MachineLearning.Checkpointing;

namespace MLFramework.Tests.Checkpointing.Metadata;

/// <summary>
/// Tests for checkpoint metadata serialization, deserialization, and validation
/// </summary>
public class CheckpointMetadataTests
{
    [Fact]
    public void CreateValidMetadata_ShouldHaveDefaultValues()
    {
        // Arrange & Act
        var metadata = new CheckpointMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal("1.0.0", metadata.Version);
        Assert.True(metadata.Timestamp > DateTime.MinValue);
        Assert.Null(metadata.Sharding);
        Assert.Null(metadata.Shards);
        Assert.Null(metadata.Training);
        Assert.Null(metadata.CustomFields);
    }

    [Fact]
    public void SerializeDeserialize_Metadata_ShouldPreserveAllFields()
    {
        // Arrange
        var originalMetadata = new CheckpointMetadata
        {
            Version = "1.2.3",
            Timestamp = new DateTime(2024, 1, 8, 12, 30, 45, DateTimeKind.Utc),
            WorldSize = 4,
            DdpRank = 0,
            Sharding = new ShardingMetadata
            {
                Strategy = "fsdp",
                ShardCount = 4,
                Precision = "fp16",
                StrategySpecificInfo = new Dictionary<string, object>
                {
                    ["wrap_policy"] = "full_shard"
                }
            },
            Shards = new List<ShardMetadata>
            {
                new ShardMetadata
                {
                    Rank = 0,
                    FilePath = "checkpoint_shard_0.shard",
                    FileSize = 1024,
                    Tensors = new List<TensorMetadata>
                    {
                        new TensorMetadata
                        {
                            Name = "layer1.weight",
                            Shape = new long[] { 512, 768 },
                            DataType = TensorDataType.Float16,
                            Offset = 0,
                            Size = 512 * 768 * 2
                        }
                    },
                    Checksum = "abc123"
                },
                new ShardMetadata
                {
                    Rank = 1,
                    FilePath = "checkpoint_shard_1.shard",
                    FileSize = 1024,
                    Tensors = new List<TensorMetadata>
                    {
                        new TensorMetadata
                        {
                            Name = "layer2.weight",
                            Shape = new long[] { 512, 768 },
                            DataType = TensorDataType.Float16,
                            Offset = 0,
                            Size = 512 * 768 * 2
                        }
                    },
                    Checksum = "def456"
                }
            },
            Training = new TrainingMetadata
            {
                Epoch = 10,
                Step = 10000,
                LearningRate = 0.001f,
                OptimizerType = "AdamW",
                OptimizerState = new Dictionary<string, object>
                {
                    ["momentum"] = 0.9f
                }
            },
            CustomFields = new Dictionary<string, string>
            {
                ["experiment_id"] = "exp_123",
                ["notes"] = "Test run"
            }
        };

        // Act
        var json = MetadataSerializer.Serialize(originalMetadata);
        var deserializedMetadata = MetadataSerializer.Deserialize(json);

        // Assert
        Assert.NotNull(deserializedMetadata);
        Assert.Equal(originalMetadata.Version, deserializedMetadata.Version);
        Assert.Equal(originalMetadata.Timestamp, deserializedMetadata.Timestamp);
        Assert.Equal(originalMetadata.WorldSize, deserializedMetadata.WorldSize);
        Assert.Equal(originalMetadata.DdpRank, deserializedMetadata.DdpRank);

        // Assert Sharding
        Assert.NotNull(deserializedMetadata.Sharding);
        Assert.Equal(originalMetadata.Sharding.Strategy, deserializedMetadata.Sharding.Strategy);
        Assert.Equal(originalMetadata.Sharding.ShardCount, deserializedMetadata.Sharding.ShardCount);
        Assert.Equal(originalMetadata.Sharding.Precision, deserializedMetadata.Sharding.Precision);
        Assert.Equal(
            originalMetadata.Sharding.StrategySpecificInfo.Count,
            deserializedMetadata.Sharding.StrategySpecificInfo.Count);

        // Assert Shards
        Assert.NotNull(deserializedMetadata.Shards);
        Assert.Equal(originalMetadata.Shards.Count, deserializedMetadata.Shards.Count);
        Assert.Equal(originalMetadata.Shards[0].Rank, deserializedMetadata.Shards[0].Rank);
        Assert.Equal(originalMetadata.Shards[0].FilePath, deserializedMetadata.Shards[0].FilePath);
        Assert.Equal(originalMetadata.Shards[0].FileSize, deserializedMetadata.Shards[0].FileSize);
        Assert.Equal(originalMetadata.Shards[0].Checksum, deserializedMetadata.Shards[0].Checksum);

        // Assert Training
        Assert.NotNull(deserializedMetadata.Training);
        Assert.Equal(originalMetadata.Training.Epoch, deserializedMetadata.Training.Epoch);
        Assert.Equal(originalMetadata.Training.Step, deserializedMetadata.Training.Step);
        Assert.Equal(originalMetadata.Training.LearningRate, deserializedMetadata.Training.LearningRate);
        Assert.Equal(originalMetadata.Training.OptimizerType, deserializedMetadata.Training.OptimizerType);

        // Assert CustomFields
        Assert.NotNull(deserializedMetadata.CustomFields);
        Assert.Equal(originalMetadata.CustomFields.Count, deserializedMetadata.CustomFields.Count);
    }

    [Fact]
    public void Validate_ValidMetadata_ShouldReturnValid()
    {
        // Arrange
        var metadata = new CheckpointMetadata
        {
            Version = "1.0.0",
            WorldSize = 2,
            DdpRank = 0,
            Sharding = new ShardingMetadata
            {
                Strategy = "ddp",
                ShardCount = 2,
                Precision = "fp32"
            },
            Shards = new List<ShardMetadata>
            {
                new ShardMetadata
                {
                    Rank = 0,
                    FilePath = "shard0.bin",
                    FileSize = 1024,
                    Tensors = new List<TensorMetadata>(),
                    Checksum = "checksum1"
                },
                new ShardMetadata
                {
                    Rank = 1,
                    FilePath = "shard1.bin",
                    FileSize = 1024,
                    Tensors = new List<TensorMetadata>(),
                    Checksum = "checksum2"
                }
            }
        };

        // Act
        var result = MetadataValidator.Validate(metadata);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public void Validate_NullMetadata_ShouldReturnInvalid()
    {
        // Arrange
        CheckpointMetadata? metadata = null;

        // Act
        var result = MetadataValidator.Validate(metadata!);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("Metadata cannot be null", result.Errors);
    }

    [Fact]
    public void Validate_MissingVersion_ShouldReturnInvalid()
    {
        // Arrange
        var metadata = new CheckpointMetadata
        {
            Version = "",
            WorldSize = 2,
            DdpRank = 0,
            Sharding = new ShardingMetadata
            {
                Strategy = "ddp",
                ShardCount = 2,
                Precision = "fp32"
            },
            Shards = new List<ShardMetadata>
            {
                new ShardMetadata { Rank = 0, FilePath = "shard0.bin", FileSize = 1024 },
                new ShardMetadata { Rank = 1, FilePath = "shard1.bin", FileSize = 1024 }
            }
        };

        // Act
        var result = MetadataValidator.Validate(metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("Version is required", result.Errors);
    }

    [Fact]
    public void Validate_MissingSharding_ShouldReturnInvalid()
    {
        // Arrange
        var metadata = new CheckpointMetadata
        {
            Version = "1.0.0",
            WorldSize = 2,
            DdpRank = 0,
            Sharding = null,
            Shards = new List<ShardMetadata>
            {
                new ShardMetadata { Rank = 0, FilePath = "shard0.bin", FileSize = 1024 }
            }
        };

        // Act
        var result = MetadataValidator.Validate(metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("Sharding metadata is required", result.Errors);
    }

    [Fact]
    public void Validate_EmptyShards_ShouldReturnInvalid()
    {
        // Arrange
        var metadata = new CheckpointMetadata
        {
            Version = "1.0.0",
            WorldSize = 2,
            DdpRank = 0,
            Sharding = new ShardingMetadata
            {
                Strategy = "ddp",
                ShardCount = 2,
                Precision = "fp32"
            },
            Shards = new List<ShardMetadata>()
        };

        // Act
        var result = MetadataValidator.Validate(metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("At least one shard is required", result.Errors);
    }

    [Fact]
    public void Validate_ShardCountMismatch_ShouldReturnInvalid()
    {
        // Arrange
        var metadata = new CheckpointMetadata
        {
            Version = "1.0.0",
            WorldSize = 4,
            DdpRank = 0,
            Sharding = new ShardingMetadata
            {
                Strategy = "ddp",
                ShardCount = 4,
                Precision = "fp32"
            },
            Shards = new List<ShardMetadata>
            {
                new ShardMetadata { Rank = 0, FilePath = "shard0.bin", FileSize = 1024 },
                new ShardMetadata { Rank = 1, FilePath = "shard1.bin", FileSize = 1024 }
            }
        };

        // Act
        var result = MetadataValidator.Validate(metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("Shard count mismatch", result.Errors);
    }

    [Fact]
    public void Validate_DuplicateRanks_ShouldReturnInvalid()
    {
        // Arrange
        var metadata = new CheckpointMetadata
        {
            Version = "1.0.0",
            WorldSize = 2,
            DdpRank = 0,
            Sharding = new ShardingMetadata
            {
                Strategy = "ddp",
                ShardCount = 2,
                Precision = "fp32"
            },
            Shards = new List<ShardMetadata>
            {
                new ShardMetadata { Rank = 0, FilePath = "shard0.bin", FileSize = 1024 },
                new ShardMetadata { Rank = 0, FilePath = "shard0_dup.bin", FileSize = 1024 }
            }
        };

        // Act
        var result = MetadataValidator.Validate(metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("Duplicate rank found: 0", result.Errors);
    }

    [Fact]
    public void Validate_MissingChecksum_ShouldReturnWarning()
    {
        // Arrange
        var metadata = new CheckpointMetadata
        {
            Version = "1.0.0",
            WorldSize = 2,
            DdpRank = 0,
            Sharding = new ShardingMetadata
            {
                Strategy = "ddp",
                ShardCount = 2,
                Precision = "fp32"
            },
            Shards = new List<ShardMetadata>
            {
                new ShardMetadata { Rank = 0, FilePath = "shard0.bin", FileSize = 1024, Checksum = null },
                new ShardMetadata { Rank = 1, FilePath = "shard1.bin", FileSize = 1024, Checksum = "" }
            }
        };

        // Act
        var result = MetadataValidator.Validate(metadata);

        // Assert
        Assert.True(result.IsValid);
        Assert.True(result.HasWarnings);
        Assert.Contains("Shard 0 missing checksum", result.Warnings);
        Assert.Contains("Shard 1 missing checksum", result.Warnings);
    }

    [Fact]
    public async Task WriteAsync_ValidMetadata_ShouldCreateFile()
    {
        // Arrange
        var metadata = new CheckpointMetadata
        {
            Version = "1.0.0",
            WorldSize = 2,
            DdpRank = 0,
            Sharding = new ShardingMetadata
            {
                Strategy = "ddp",
                ShardCount = 2,
                Precision = "fp32"
            },
            Shards = new List<ShardMetadata>
            {
                new ShardMetadata { Rank = 0, FilePath = "shard0.bin", FileSize = 1024 }
            }
        };
        var tempPath = Path.Combine(Path.GetTempPath(), $"test_metadata_{Guid.NewGuid()}.json");

        try
        {
            // Act
            await MetadataSerializer.WriteAsync(metadata, tempPath);

            // Assert
            Assert.True(File.Exists(tempPath));
            var content = await File.ReadAllTextAsync(tempPath);
            Assert.Contains("1.0.0", content);
        }
        finally
        {
            // Cleanup
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    [Fact]
    public async Task ReadAsync_ExistingFile_ShouldReturnMetadata()
    {
        // Arrange
        var metadata = new CheckpointMetadata
        {
            Version = "1.0.0",
            WorldSize = 2,
            DdpRank = 0,
            Sharding = new ShardingMetadata
            {
                Strategy = "ddp",
                ShardCount = 2,
                Precision = "fp32"
            },
            Shards = new List<ShardMetadata>
            {
                new ShardMetadata { Rank = 0, FilePath = "shard0.bin", FileSize = 1024 }
            }
        };
        var tempPath = Path.Combine(Path.GetTempPath(), $"test_metadata_{Guid.NewGuid()}.json");

        try
        {
            // Act
            await MetadataSerializer.WriteAsync(metadata, tempPath);
            var readMetadata = await MetadataSerializer.ReadAsync(tempPath);

            // Assert
            Assert.NotNull(readMetadata);
            Assert.Equal(metadata.Version, readMetadata.Version);
            Assert.Equal(metadata.WorldSize, readMetadata.WorldSize);
            Assert.Equal(metadata.Sharding.Strategy, readMetadata.Sharding.Strategy);
            Assert.Equal(metadata.Shards.Count, readMetadata.Shards.Count);
        }
        finally
        {
            // Cleanup
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    [Fact]
    public async Task ReadAsync_NonExistentFile_ShouldThrowFileNotFoundException()
    {
        // Arrange
        var nonExistentPath = Path.Combine(Path.GetTempPath(), $"nonexistent_{Guid.NewGuid()}.json");

        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            MetadataSerializer.ReadAsync(nonExistentPath));
    }

    [Fact]
    public void Serialize_WithNullMetadata_ShouldThrowArgumentNullException()
    {
        // Arrange
        CheckpointMetadata? metadata = null;

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            MetadataSerializer.Serialize(metadata!));
    }

    [Fact]
    public void Deserialize_WithEmptyJson_ShouldThrowArgumentException()
    {
        // Arrange
        string emptyJson = "";

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            MetadataSerializer.Deserialize(emptyJson));
    }

    [Fact]
    public void Deserialize_WithInvalidJson_ShouldThrowInvalidOperationException()
    {
        // Arrange
        string invalidJson = "{ invalid json }";

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            MetadataSerializer.Deserialize(invalidJson));
    }

    [Fact]
    public void ValidateOrThrow_WithInvalidMetadata_ShouldThrowInvalidOperationException()
    {
        // Arrange
        var metadata = new CheckpointMetadata
        {
            Version = "",
            Sharding = null,
            Shards = new List<ShardMetadata>()
        };

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            MetadataValidator.ValidateOrThrow(metadata));
    }

    [Fact]
    public void ValidateOrThrow_WithValidMetadata_ShouldNotThrow()
    {
        // Arrange
        var metadata = new CheckpointMetadata
        {
            Version = "1.0.0",
            WorldSize = 2,
            DdpRank = 0,
            Sharding = new ShardingMetadata
            {
                Strategy = "ddp",
                ShardCount = 2,
                Precision = "fp32"
            },
            Shards = new List<ShardMetadata>
            {
                new ShardMetadata { Rank = 0, FilePath = "shard0.bin", FileSize = 1024 },
                new ShardMetadata { Rank = 1, FilePath = "shard1.bin", FileSize = 1024 }
            }
        };

        // Act & Assert - should not throw
        MetadataValidator.ValidateOrThrow(metadata);
    }
}
