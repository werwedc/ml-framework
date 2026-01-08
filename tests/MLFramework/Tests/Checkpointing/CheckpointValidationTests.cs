namespace MachineLearning.Checkpointing.Tests;

using System.Security.Cryptography;
using Xunit;

/// <summary>
/// Tests for checkpoint validation components
/// </summary>
public class CheckpointValidationTests
{
    [Fact]
    public void ValidationResult_AddSubResults_MergesErrorsAndWarnings()
    {
        // Arrange
        var result = new ValidationResult();
        var subResult = new ValidationResult();
        subResult.AddError("Sub error 1");
        subResult.AddWarning("Sub warning 1");

        // Act
        result.AddSubResults(subResult);

        // Assert
        Assert.True(result.Errors.Any(e => e.Contains("Sub error 1")));
        Assert.True(result.Warnings.Any(w => w.Contains("Sub warning 1")));
    }

    [Fact]
    public void ValidationResult_IsValid_WhenSubResultsHaveErrors_ReturnsFalse()
    {
        // Arrange
        var result = new ValidationResult();
        var subResult = new ValidationResult();
        subResult.AddError("Sub error");

        // Act
        result.AddSubResults(subResult);

        // Assert
        Assert.False(result.IsValid);
    }

    [Fact]
    public void ValidationResult_GetSummary_FormatsCorrectly()
    {
        // Arrange
        var result = new ValidationResult();
        result.AddError("Test error");
        result.AddWarning("Test warning");

        // Act
        var summary = result.GetSummary();

        // Assert
        Assert.Contains("INVALID", summary);
        Assert.Contains("Test error", summary);
        Assert.Contains("Test warning", summary);
    }
}

/// <summary>
/// Tests for ChecksumIntegrityChecker
/// </summary>
public class ChecksumIntegrityCheckerTests
{
    [Fact]
    public async Task CheckIntegrityAsync_WithMatchingChecksum_ReturnsValid()
    {
        // Arrange
        var checker = new ChecksumIntegrityChecker();
        var data = System.Text.Encoding.UTF8.GetBytes("test data");
        var checksum = ComputeChecksum(data);
        var shardMeta = new ShardMetadata
        {
            Rank = 0,
            Checksum = checksum
        };

        // Act
        var result = await checker.CheckIntegrityAsync(data, shardMeta);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public async Task CheckIntegrityAsync_WithMismatchingChecksum_ReturnsInvalid()
    {
        // Arrange
        var checker = new ChecksumIntegrityChecker();
        var data = System.Text.Encoding.UTF8.GetBytes("test data");
        var shardMeta = new ShardMetadata
        {
            Rank = 0,
            Checksum = "invalid_checksum"
        };

        // Act
        var result = await checker.CheckIntegrityAsync(data, shardMeta);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("checksum", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task CheckIntegrityAsync_WithMissingChecksum_ReturnsWarning()
    {
        // Arrange
        var checker = new ChecksumIntegrityChecker();
        var data = System.Text.Encoding.UTF8.GetBytes("test data");
        var shardMeta = new ShardMetadata
        {
            Rank = 0,
            Checksum = null
        };

        // Act
        var result = await checker.CheckIntegrityAsync(data, shardMeta);

        // Assert
        Assert.True(result.IsValid); // No error, but has warning
        Assert.True(result.HasWarnings);
    }

    private string ComputeChecksum(byte[] data)
    {
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(data);
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
    }
}

/// <summary>
/// Tests for SizeIntegrityChecker
/// </summary>
public class SizeIntegrityCheckerTests
{
    [Fact]
    public async Task CheckIntegrityAsync_WithMatchingSize_ReturnsValid()
    {
        // Arrange
        var checker = new SizeIntegrityChecker();
        var data = System.Text.Encoding.UTF8.GetBytes("test data");
        var shardMeta = new ShardMetadata
        {
            Rank = 0,
            FileSize = data.Length
        };

        // Act
        var result = await checker.CheckIntegrityAsync(data, shardMeta);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public async Task CheckIntegrityAsync_WithMismatchingSize_ReturnsInvalid()
    {
        // Arrange
        var checker = new SizeIntegrityChecker();
        var data = System.Text.Encoding.UTF8.GetBytes("test data");
        var shardMeta = new ShardMetadata
        {
            Rank = 0,
            FileSize = 999 // Wrong size
        };

        // Act
        var result = await checker.CheckIntegrityAsync(data, shardMeta);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("size", StringComparison.OrdinalIgnoreCase));
    }
}

/// <summary>
/// Tests for VersionCompatibilityChecker
/// </summary>
public class VersionCompatibilityCheckerTests
{
    [Fact]
    public async Task CheckCompatibilityAsync_WithCompatibleVersion_ReturnsValid()
    {
        // Arrange
        var checker = new VersionCompatibilityChecker("1.0.x");
        var metadata = new CheckpointMetadata
        {
            Version = "1.0.5"
        };

        // Act
        var result = await checker.CheckCompatibilityAsync(metadata);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public async Task CheckCompatibilityAsync_WithIncompatibleVersion_ReturnsInvalid()
    {
        // Arrange
        var checker = new VersionCompatibilityChecker("1.0.x");
        var metadata = new CheckpointMetadata
        {
            Version = "2.0.0"
        };

        // Act
        var result = await checker.CheckCompatibilityAsync(metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("version", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task CheckCompatibilityAsync_WithInvalidVersionFormat_ReturnsInvalid()
    {
        // Arrange
        var checker = new VersionCompatibilityChecker("1.0.x");
        var metadata = new CheckpointMetadata
        {
            Version = "invalid_version"
        };

        // Act
        var result = await checker.CheckCompatibilityAsync(metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("Invalid version", StringComparison.OrdinalIgnoreCase));
    }
}

/// <summary>
/// Tests for SchemaCompatibilityChecker
/// </summary>
public class SchemaCompatibilityCheckerTests
{
    [Fact]
    public async Task CheckCompatibilityAsync_WithValidSchema_ReturnsValid()
    {
        // Arrange
        var checker = new SchemaCompatibilityChecker();
        var metadata = new CheckpointMetadata
        {
            Sharding = new ShardingMetadata
            {
                Strategy = "fsdp",
                Precision = "fp16"
            }
        };

        // Act
        var result = await checker.CheckCompatibilityAsync(metadata);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public async Task CheckCompatibilityAsync_WithUnknownStrategy_ReturnsWarning()
    {
        // Arrange
        var checker = new SchemaCompatibilityChecker();
        var metadata = new CheckpointMetadata
        {
            Sharding = new ShardingMetadata
            {
                Strategy = "unknown_strategy",
                Precision = "fp16"
            }
        };

        // Act
        var result = await checker.CheckCompatibilityAsync(metadata);

        // Assert
        Assert.True(result.IsValid); // Warning is not an error
        Assert.True(result.HasWarnings);
        Assert.Contains(result.Warnings, w => w.Contains("strategy", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task CheckCompatibilityAsync_WithUnknownPrecision_ReturnsWarning()
    {
        // Arrange
        var checker = new SchemaCompatibilityChecker();
        var metadata = new CheckpointMetadata
        {
            Sharding = new ShardingMetadata
            {
                Strategy = "fsdp",
                Precision = "unknown_precision"
            }
        };

        // Act
        var result = await checker.CheckCompatibilityAsync(metadata);

        // Assert
        Assert.True(result.IsValid); // Warning is not an error
        Assert.True(result.HasWarnings);
        Assert.Contains(result.Warnings, w => w.Contains("precision", StringComparison.OrdinalIgnoreCase));
    }
}

/// <summary>
/// Tests for CheckpointValidator metadata validation
/// </summary>
public class CheckpointValidatorMetadataTests
{
    [Fact]
    public async Task ValidateMetadataAsync_WithValidMetadata_ReturnsValid()
    {
        // Arrange
        var storage = new MockCheckpointStorage();
        var validator = new CheckpointValidator(storage);
        var metadata = new CheckpointMetadata
        {
            Version = "1.0.0",
            Sharding = new ShardingMetadata
            {
                Strategy = "fsdp",
                ShardCount = 2,
                Precision = "fp16"
            },
            Shards = new List<ShardMetadata>
            {
                new ShardMetadata { Rank = 0, FilePath = "shard_0.shard", FileSize = 100, Checksum = "abc123" },
                new ShardMetadata { Rank = 1, FilePath = "shard_1.shard", FileSize = 100, Checksum = "def456" }
            }
        };

        // Act
        var result = await validator.ValidateMetadataAsync(metadata);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public async Task ValidateMetadataAsync_WithMissingVersion_ReturnsInvalid()
    {
        // Arrange
        var storage = new MockCheckpointStorage();
        var validator = new CheckpointValidator(storage);
        var metadata = new CheckpointMetadata
        {
            Version = "",
            Sharding = new ShardingMetadata
            {
                Strategy = "fsdp",
                ShardCount = 2,
                Precision = "fp16"
            },
            Shards = new List<ShardMetadata>
            {
                new ShardMetadata { Rank = 0, FilePath = "shard_0.shard", FileSize = 100 }
            }
        };

        // Act
        var result = await validator.ValidateMetadataAsync(metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("version", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task ValidateMetadataAsync_WithMissingSharding_ReturnsInvalid()
    {
        // Arrange
        var storage = new MockCheckpointStorage();
        var validator = new CheckpointValidator(storage);
        var metadata = new CheckpointMetadata
        {
            Version = "1.0.0",
            Sharding = null,
            Shards = new List<ShardMetadata>
            {
                new ShardMetadata { Rank = 0, FilePath = "shard_0.shard", FileSize = 100 }
            }
        };

        // Act
        var result = await validator.ValidateMetadataAsync(metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("sharding", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task ValidateMetadataAsync_WithShardCountMismatch_ReturnsInvalid()
    {
        // Arrange
        var storage = new MockCheckpointStorage();
        var validator = new CheckpointValidator(storage);
        var metadata = new CheckpointMetadata
        {
            Version = "1.0.0",
            Sharding = new ShardingMetadata
            {
                Strategy = "fsdp",
                ShardCount = 2, // Expects 2 shards
                Precision = "fp16"
            },
            Shards = new List<ShardMetadata>
            {
                new ShardMetadata { Rank = 0, FilePath = "shard_0.shard", FileSize = 100 }
                // Only 1 shard provided
            }
        };

        // Act
        var result = await validator.ValidateMetadataAsync(metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("Shard count", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task ValidateMetadataAsync_WithDuplicateRanks_ReturnsInvalid()
    {
        // Arrange
        var storage = new MockCheckpointStorage();
        var validator = new CheckpointValidator(storage);
        var metadata = new CheckpointMetadata
        {
            Version = "1.0.0",
            Sharding = new ShardingMetadata
            {
                Strategy = "fsdp",
                ShardCount = 3,
                Precision = "fp16"
            },
            Shards = new List<ShardMetadata>
            {
                new ShardMetadata { Rank = 0, FilePath = "shard_0.shard", FileSize = 100 },
                new ShardMetadata { Rank = 1, FilePath = "shard_1.shard", FileSize = 100 },
                new ShardMetadata { Rank = 1, FilePath = "shard_1_dup.shard", FileSize = 100 } // Duplicate rank
            }
        };

        // Act
        var result = await validator.ValidateMetadataAsync(metadata);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("Duplicate", StringComparison.OrdinalIgnoreCase));
    }
}

/// <summary>
/// Mock checkpoint storage for testing
/// </summary>
public class MockCheckpointStorage : ICheckpointStorage
{
    private readonly Dictionary<string, byte[]> _storage = new();

    public void AddFile(string path, byte[] data)
    {
        _storage[path] = data;
    }

    public Task<bool> ExistsAsync(string path, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(_storage.ContainsKey(path));
    }

    public Task<byte[]> ReadAsync(string path, CancellationToken cancellationToken = default)
    {
        if (_storage.TryGetValue(path, out var data))
        {
            return Task.FromResult(data);
        }
        throw new FileNotFoundException($"File not found: {path}");
    }

    public Task WriteAsync(string path, byte[] data, CancellationToken cancellationToken = default)
    {
        _storage[path] = data;
        return Task.CompletedTask;
    }

    public Task DeleteAsync(string path, CancellationToken cancellationToken = default)
    {
        _storage.Remove(path);
        return Task.CompletedTask;
    }
}
