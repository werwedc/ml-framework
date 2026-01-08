namespace MachineLearning.Checkpointing.Tests;

using System.Security.Cryptography;
using Xunit;

/// <summary>
/// Tests for ChecksumIntegrityChecker
/// </summary>
public class ChecksumIntegrityCheckerTests
{
    private ChecksumIntegrityChecker _checker = null!;

    public ChecksumIntegrityCheckerTests()
    {
        _checker = new ChecksumIntegrityChecker();
    }

    [Fact]
    public async Task CheckIntegrityAsync_WithMatchingChecksum_ReturnsValid()
    {
        // Arrange
        var data = System.Text.Encoding.UTF8.GetBytes("test data");
        var checksum = ComputeChecksum(data);
        var shardMeta = new ShardMetadata
        {
            Rank = 0,
            Checksum = checksum
        };

        // Act
        var result = await _checker.CheckIntegrityAsync(data, shardMeta);

        // Assert
        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public async Task CheckIntegrityAsync_WithMismatchingChecksum_ReturnsInvalid()
    {
        // Arrange
        var data = System.Text.Encoding.UTF8.GetBytes("test data");
        var shardMeta = new ShardMetadata
        {
            Rank = 0,
            Checksum = "invalid_checksum"
        };

        // Act
        var result = await _checker.CheckIntegrityAsync(data, shardMeta);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("checksum", StringComparison.OrdinalIgnoreCase));
    }

    private string ComputeChecksum(byte[] data)
    {
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(data);
        return BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
    }
}
