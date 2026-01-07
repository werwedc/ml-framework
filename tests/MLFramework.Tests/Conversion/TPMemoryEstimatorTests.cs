using Xunit;
using MLFramework.Conversion;
using MLFramework.Modules;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Conversion;

/// <summary>
/// Tests for TPMemoryEstimator.
/// </summary>
public class TPMemoryEstimatorTests
{
    [Fact]
    public void EstimateMemory_SimpleLinearLayer_CalculatesCorrectly()
    {
        // Arrange
        var linear = new Linear(inFeatures: 128, outFeatures: 256, useBias: true);
        int worldSize = 4;

        // Act
        var estimate = TPMemoryEstimator.EstimateMemory(linear, worldSize);

        // Assert
        Assert.NotNull(estimate);
        Assert.True(estimate.BaseMemoryMB > 0);
    }

    [Fact]
    public void EstimateMemory_BaseMemory_IsCalculatedCorrectly()
    {
        // Arrange
        int inFeatures = 128;
        int outFeatures = 256;
        var linear = new Linear(inFeatures, outFeatures, useBias: true);

        // Calculate expected base memory
        // Weight: 256 * 128 * 4 bytes = 131,072 bytes = ~128 KB
        // Bias: 256 * 4 bytes = 1,024 bytes = ~1 KB
        long expectedWeightMemory = outFeatures * inFeatures * 4;
        long expectedBiasMemory = outFeatures * 4;
        long expectedTotalBytes = expectedWeightMemory + expectedBiasMemory;
        long expectedMB = expectedTotalBytes / 1024 / 1024;

        // Act
        var estimate = TPMemoryEstimator.EstimateMemory(linear, worldSize: 2);

        // Assert
        Assert.Equal(expectedMB, estimate.BaseMemoryMB);
    }

    [Fact]
    public void EstimateMemory_TPMemoryPerRank_IsLowerThanBase()
    {
        // Arrange
        var linear = new Linear(inFeatures: 1024, outFeatures: 4096, useBias: false);
        int worldSize = 4;

        // Act
        var estimate = TPMemoryEstimator.EstimateMemory(linear, worldSize);

        // Assert
        Assert.True(estimate.TPMemoryPerRankMB < estimate.BaseMemoryMB);
    }

    [Fact]
    public void EstimateMemory_HigherWorldSize_LowersMemoryPerRank()
    {
        // Arrange
        var linear = new Linear(inFeatures: 1024, outFeatures: 4096, useBias: false);

        // Act
        var estimate2 = TPMemoryEstimator.EstimateMemory(linear, worldSize: 2);
        var estimate4 = TPMemoryEstimator.EstimateMemory(linear, worldSize: 4);

        // Assert
        Assert.True(estimate4.TPMemoryPerRankMB < estimate2.TPMemoryPerRankMB);
    }

    [Fact]
    public void EstimateMemory_CommunicationOverhead_IsPositive()
    {
        // Arrange
        var linear = new Linear(inFeatures: 512, outFeatures: 1024, useBias: true);
        int worldSize = 4;

        // Act
        var estimate = TPMemoryEstimator.EstimateMemory(linear, worldSize);

        // Assert
        Assert.True(estimate.CommunicationOverheadMB > 0);
    }

    [Fact]
    public void EstimateMemory_CommunicationOverhead_IsTenPercent()
    {
        // Arrange
        var linear = new Linear(inFeatures: 256, outFeatures: 512, useBias: false);
        int worldSize = 4;

        // Act
        var estimate = TPMemoryEstimator.EstimateMemory(linear, worldSize);

        // Assert - Should be approximately 10%
        long expectedOverhead = (long)(estimate.TPMemoryPerRankMB * 0.1);
        Assert.Equal(expectedOverhead, estimate.CommunicationOverheadMB);
    }

    [Fact]
    public void EstimateMemory_TotalMemoryPerRank_IncludesOverhead()
    {
        // Arrange
        var linear = new Linear(inFeatures: 512, outFeatures: 1024, useBias: true);
        int worldSize = 4;

        // Act
        var estimate = TPMemoryEstimator.EstimateMemory(linear, worldSize);

        // Assert
        Assert.Equal(
            estimate.TPMemoryPerRankMB + estimate.CommunicationOverheadMB,
            estimate.TotalMemoryPerRankMB);
    }

    [Fact]
    public void EstimateMemory_MemorySavingsPercentage_IsPositiveWhenParallelizable()
    {
        // Arrange
        var linear = new Linear(inFeatures: 1024, outFeatures: 4096, useBias: false);
        int worldSize = 4;

        // Act
        var estimate = TPMemoryEstimator.EstimateMemory(linear, worldSize);

        // Assert
        Assert.True(estimate.MemorySavingsPercentage > 0);
    }

    [Fact]
    public void EstimateMemory_MemorySavingsPercentage_ZeroWhenNotParallelizable()
    {
        // Arrange - Small balanced layer that won't be parallelized
        var linear = new Linear(inFeatures: 32, outFeatures: 32, useBias: false);
        int worldSize = 4;

        // Act
        var estimate = TPMemoryEstimator.EstimateMemory(linear, worldSize);

        // Assert
        Assert.Equal(0.0, estimate.MemorySavingsPercentage);
    }

    [Fact]
    public void EstimateMemory_MemorySavingsPercentage_ScalesWithWorldSize()
    {
        // Arrange
        var linear = new Linear(inFeatures: 1024, outFeatures: 4096, useBias: false);

        // Act
        var estimate2 = TPMemoryEstimator.EstimateMemory(linear, worldSize: 2);
        var estimate4 = TPMemoryEstimator.EstimateMemory(linear, worldSize: 4);
        var estimate8 = TPMemoryEstimator.EstimateMemory(linear, worldSize: 8);

        // Assert
        Assert.True(estimate8.MemorySavingsPercentage > estimate4.MemorySavingsPercentage);
        Assert.True(estimate4.MemorySavingsPercentage > estimate2.MemorySavingsPercentage);
    }

    [Fact]
    public void EstimateMemory_Conv2dLayer_CalculatesCorrectly()
    {
        // Arrange
        var conv = new Conv2d(
            inChannels: 64,
            outChannels: 256,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            useBias: true);
        int worldSize = 4;

        // Act
        var estimate = TPMemoryEstimator.EstimateMemory(conv, worldSize);

        // Assert
        Assert.NotNull(estimate);
        Assert.True(estimate.BaseMemoryMB > 0);
    }

    [Fact]
    public void EstimateMemory_GenerateSummary_ProducesValidOutput()
    {
        // Arrange
        var linear = new Linear(inFeatures: 256, outFeatures: 512, useBias: true);
        int worldSize = 4;
        var estimate = TPMemoryEstimator.EstimateMemory(linear, worldSize);

        // Act
        var summary = estimate.GenerateSummary();

        // Assert
        Assert.NotNull(summary);
        Assert.Contains("Base memory:", summary);
        Assert.Contains("TP memory per rank:", summary);
        Assert.Contains("Savings:", summary);
        Assert.Contains("Communication overhead:", summary);
    }

    [Fact]
    public void EstimateMemory_MemorySavingsPercentage_NeverExceeds100()
    {
        // Arrange
        var linear = new Linear(inFeatures: 2048, outFeatures: 8192, useBias: false);
        int worldSize = 2;

        // Act
        var estimate = TPMemoryEstimator.EstimateMemory(linear, worldSize);

        // Assert
        Assert.True(estimate.MemorySavingsPercentage <= 100);
    }

    [Fact]
    public void EstimateMemory_HighlyParallelizableModel_SignificantSavings()
    {
        // Arrange - Large model that should benefit greatly from TP
        var linear = new Linear(inFeatures: 4096, outFeatures: 16384, useBias: false);
        int worldSize = 8;

        // Act
        var estimate = TPMemoryEstimator.EstimateMemory(linear, worldSize);

        // Assert
        Assert.True(estimate.MemorySavingsPercentage > 50);
    }
}
