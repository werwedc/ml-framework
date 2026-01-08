namespace MLFramework.Tests.Checkpointing.Formats;

using MachineLearning.Checkpointing;
using System.Security.Cryptography;
using System.Text;

/// <summary>
/// Tests for the shard file verifier
/// </summary>
public class ShardFileVerifierTests
{
    private readonly ShardFileVerifier _verifier;
    private readonly MemoryCheckpointStorage _storage;
    private readonly MultiShardCheckpointFormat _format;
    private readonly MultiShardSaver _saver;

    public ShardFileVerifierTests()
    {
        _storage = new MemoryCheckpointStorage();
        _verifier = new ShardFileVerifier(_storage);
        _format = new MultiShardCheckpointFormat();
        _saver = new MultiShardSaver(_format, _storage);
    }

    [Fact]
    public void Constructor_NullStorage_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new ShardFileVerifier(null));
    }

    [Fact]
    public async Task VerifyCheckpointAsync_AllShardsValid_ReturnsValidResult()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_valid";
        var metadata = CreateTestMetadata(3);
        var shardData = new List<ShardData>();

        // Save all shards with valid checksums
        for (int i = 0; i < 3; i++)
        {
            var shard = CreateTestShard(i);
            var shardMeta = CreateTestShardMetadata(i);
            var serialized = await _format.SerializeShardAsync(shard, shardMeta);
            shardMeta.Checksum = ComputeChecksum(serialized);
            await _saver.SaveShardAsync(checkpointPrefix, i, shard, shardMeta);
            shardData.Add(shard);
        }

        // Save metadata
        await _saver.SaveMetadataAsync(checkpointPrefix, metadata);

        // Act
        var result = await _verifier.VerifyCheckpointAsync(checkpointPrefix, metadata);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
        Assert.Empty(result.Warnings);
    }

    [Fact]
    public async Task VerifyCheckpointAsync_MissingShard_ReturnsInvalidResult()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_missing";
        var metadata = CreateTestMetadata(3);

        // Save only 2 shards (missing one)
        for (int i = 0; i < 2; i++)
        {
            var shard = CreateTestShard(i);
            var shardMeta = CreateTestShardMetadata(i);
            await _saver.SaveShardAsync(checkpointPrefix, i, shard, shardMeta);
        }

        // Save metadata
        await _saver.SaveMetadataAsync(checkpointPrefix, metadata);

        // Act
        var result = await _verifier.VerifyCheckpointAsync(checkpointPrefix, metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.NotEmpty(result.Errors);
        Assert.Contains(result.Errors, e => e.Contains("Shard file not found"));
    }

    [Fact]
    public async Task VerifyCheckpointAsync_MissingMetadata_ReturnsInvalidResult()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_no_meta";
        var metadata = CreateTestMetadata(2);

        // Save shards but not metadata
        for (int i = 0; i < 2; i++)
        {
            var shard = CreateTestShard(i);
            var shardMeta = CreateTestShardMetadata(i);
            await _saver.SaveShardAsync(checkpointPrefix, i, shard, shardMeta);
        }

        // Act
        var result = await _verifier.VerifyCheckpointAsync(checkpointPrefix, metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("Metadata file not found"));
    }

    [Fact]
    public async Task VerifyCheckpointAsync_FileSizeMismatch_ReturnsInvalidResult()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_size";
        var metadata = CreateTestMetadata(2);

        // Save shard
        var shard = CreateTestShard(0);
        var shardMeta = CreateTestShardMetadata(0);
        await _saver.SaveShardAsync(checkpointPrefix, 0, shard, shardMeta);

        // Update metadata with wrong file size
        shardMeta.FileSize = 99999; // Wrong size

        // Act
        var result = await _verifier.VerifyCheckpointAsync(checkpointPrefix, metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("Shard file size mismatch"));
    }

    [Fact]
    public async Task VerifyCheckpointAsync_ChecksumMismatch_ReturnsInvalidResult()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_checksum";
        var metadata = CreateTestMetadata(2);

        // Save shard
        var shard = CreateTestShard(0);
        var shardMeta = CreateTestShardMetadata(0);
        shardMeta.Checksum = "wrong_checksum_value";
        await _saver.SaveShardAsync(checkpointPrefix, 0, shard, shardMeta);

        // Act
        var result = await _verifier.VerifyCheckpointAsync(checkpointPrefix, metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("Shard checksum mismatch"));
    }

    [Fact]
    public async Task VerifyCheckpointAsync_EmptyPrefix_ThrowsArgumentException()
    {
        // Arrange
        var metadata = CreateTestMetadata(2);

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _verifier.VerifyCheckpointAsync("", metadata));
    }

    [Fact]
    public async Task VerifyCheckpointAsync_NullMetadata_ThrowsArgumentNullException()
    {
        // Act & Assert
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _verifier.VerifyCheckpointAsync("checkpoint", null));
    }

    [Fact]
    public async Task VerifyCheckpointAsync_MultipleErrors_ReturnsAllErrors()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_multiple_errors";
        var metadata = CreateTestMetadata(4);

        // Save only 1 shard (3 missing)
        var shard = CreateTestShard(0);
        var shardMeta = CreateTestShardMetadata(0);
        shardMeta.Checksum = "wrong_checksum";
        await _saver.SaveShardAsync(checkpointPrefix, 0, shard, shardMeta);

        // Save metadata
        await _saver.SaveMetadataAsync(checkpointPrefix, metadata);

        // Act
        var result = await _verifier.VerifyCheckpointAsync(checkpointPrefix, metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.True(result.Errors.Count >= 3); // At least 3 missing shards + 1 checksum mismatch
    }

    [Fact]
    public async Task VerifyCheckpointAsync_NoShardsMetadata_ReturnsValidResult()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_no_shards";
        var metadata = CreateTestMetadata(0);

        // Save metadata only
        await _saver.SaveMetadataAsync(checkpointPrefix, metadata);

        // Act
        var result = await _verifier.VerifyCheckpointAsync(checkpointPrefix, metadata);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public async Task VerifyCheckpointAsync_NullShardList_ReturnsValidResult()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_null_shards";
        var metadata = CreateTestMetadata(0);
        metadata.Shards = null;

        // Save metadata only
        await _saver.SaveMetadataAsync(checkpointPrefix, metadata);

        // Act
        var result = await _verifier.VerifyCheckpointAsync(checkpointPrefix, metadata);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public async Task VerifyCheckpointAsync_ValidChecksum_PassesVerification()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_valid_checksum";
        var metadata = CreateTestMetadata(2);

        // Save shard with correct checksum
        var shard = CreateTestShard(0);
        var shardMeta = CreateTestShardMetadata(0);
        var serialized = await _format.SerializeShardAsync(shard, shardMeta);
        shardMeta.Checksum = ComputeChecksum(serialized);
        await _saver.SaveShardAsync(checkpointPrefix, 0, shard, shardMeta);

        // Save metadata
        await _saver.SaveMetadataAsync(checkpointPrefix, metadata);

        // Act
        var result = await _verifier.VerifyCheckpointAsync(checkpointPrefix, metadata);

        // Assert
        Assert.True(result.IsValid);
        Assert.DoesNotContain(result.Errors, e => e.Contains("checksum"));
    }

    [Fact]
    public async Task VerifyCheckpointAsync_EmptyChecksum_DoesNotVerify()
    {
        // Arrange
        var checkpointPrefix = "checkpoint_empty_checksum";
        var metadata = CreateTestMetadata(2);

        // Save shard without checksum
        var shard = CreateTestShard(0);
        var shardMeta = CreateTestShardMetadata(0);
        shardMeta.Checksum = null; // No checksum
        await _saver.SaveShardAsync(checkpointPrefix, 0, shard, shardMeta);

        // Save metadata
        await _saver.SaveMetadataAsync(checkpointPrefix, metadata);

        // Act
        var result = await _verifier.VerifyCheckpointAsync(checkpointPrefix, metadata);

        // Assert
        Assert.True(result.IsValid);
        Assert.DoesNotContain(result.Errors, e => e.Contains("checksum"));
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

    private string ComputeChecksum(byte[] data)
    {
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(data);
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
    }
}

/// <summary>
/// VerificationResult tests
/// </summary>
public class VerificationResultTests
{
    [Fact]
    public void VerificationResult_Constructor_InitializesEmptyCollections()
    {
        // Arrange & Act
        var result = new VerificationResult();

        // Assert
        Assert.NotNull(result.Errors);
        Assert.NotNull(result.Warnings);
        Assert.Empty(result.Errors);
        Assert.Empty(result.Warnings);
    }

    [Fact]
    public void VerificationResult_IsValid_WithNoErrors_ReturnsTrue()
    {
        // Arrange
        var result = new VerificationResult();

        // Act & Assert
        Assert.True(result.IsValid);
    }

    [Fact]
    public void VerificationResult_IsValid_WithErrors_ReturnsFalse()
    {
        // Arrange
        var result = new VerificationResult();
        result.AddError("Test error");

        // Act & Assert
        Assert.False(result.IsValid);
    }

    [Fact]
    public void VerificationResult_AddError_AddsToErrorsList()
    {
        // Arrange
        var result = new VerificationResult();

        // Act
        result.AddError("Error 1");
        result.AddError("Error 2");

        // Assert
        Assert.Equal(2, result.Errors.Count);
        Assert.Equal("Error 1", result.Errors[0]);
        Assert.Equal("Error 2", result.Errors[1]);
    }

    [Fact]
    public void VerificationResult_AddWarning_AddsToWarningsList()
    {
        // Arrange
        var result = new VerificationResult();

        // Act
        result.AddWarning("Warning 1");
        result.AddWarning("Warning 2");

        // Assert
        Assert.Equal(2, result.Warnings.Count);
        Assert.Equal("Warning 1", result.Warnings[0]);
        Assert.Equal("Warning 2", result.Warnings[1]);
    }

    [Fact]
    public void VerificationResult_AddError_WithEmptyOrWhitespace_DoesNotAdd()
    {
        // Arrange
        var result = new VerificationResult();

        // Act
        result.AddError("");
        result.AddError("   ");
        result.AddError(null);

        // Assert
        Assert.Empty(result.Errors);
    }

    [Fact]
    public void VerificationResult_AddWarning_WithEmptyOrWhitespace_DoesNotAdd()
    {
        // Arrange
        var result = new VerificationResult();

        // Act
        result.AddWarning("");
        result.AddWarning("   ");
        result.AddWarning(null);

        // Assert
        Assert.Empty(result.Warnings);
    }
}
