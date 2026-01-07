using Xunit;
using MLFramework.Conversion;
using MLFramework.Modules;
using RitterFramework.Core.Tensor;

namespace MLFramework.Tests.Conversion;

/// <summary>
/// Tests for TPModelAnalyzer.
/// </summary>
public class TPModelAnalyzerTests
{
    [Fact]
    public void Analyze_SimpleLinearLayer_ReportsCorrectly()
    {
        // Arrange
        var linear = new Linear(inFeatures: 128, outFeatures: 512, useBias: true);

        // Act
        var report = TPModelAnalyzer.Analyze(linear, maxWorldSize: 4);

        // Assert
        Assert.NotNull(report);
        Assert.True(report.Layers.Count > 0);
        Assert.True(report.TotalMemoryBytes > 0);
    }

    [Fact]
    public void Analyze_LargeOutputDimension_SuggestsColumnParallelism()
    {
        // Arrange - Create a layer with large output dimension
        var linear = new Linear(inFeatures: 128, outFeatures: 1024, useBias: false);

        // Act
        var report = TPModelAnalyzer.Analyze(linear, maxWorldSize: 4);

        // Assert
        var weightLayer = report.Layers.Find(l => l.LayerName.EndsWith(".weight"));
        Assert.NotNull(weightLayer);
        Assert.True(weightLayer.IsParallelizable);
        Assert.Equal(ParallelismType.Column, weightLayer.SuggestedParallelism);
    }

    [Fact]
    public void Analyze_LargeInputDimension_SuggestsRowParallelism()
    {
        // Arrange - Create a layer with large input dimension
        var linear = new Linear(inFeatures: 1024, outFeatures: 128, useBias: false);

        // Act
        var report = TPModelAnalyzer.Analyze(linear, maxWorldSize: 4);

        // Assert
        var weightLayer = report.Layers.Find(l => l.LayerName.EndsWith(".weight"));
        Assert.NotNull(weightLayer);
        Assert.True(weightLayer.IsParallelizable);
        Assert.Equal(ParallelismType.Row, weightLayer.SuggestedParallelism);
    }

    [Fact]
    public void Analyze_BalancedDimensions_DoesNotSuggestParallelism()
    {
        // Arrange - Create a layer with balanced dimensions
        var linear = new Linear(inFeatures: 256, outFeatures: 256, useBias: false);

        // Act
        var report = TPModelAnalyzer.Analyze(linear, maxWorldSize: 4);

        // Assert
        var weightLayer = report.Layers.Find(l => l.LayerName.EndsWith(".weight"));
        Assert.NotNull(weightLayer);
        Assert.False(weightLayer.IsParallelizable);
    }

    [Fact]
    public void Analyze_Conv2dLayer_ReportsCorrectly()
    {
        // Arrange
        var conv = new Conv2d(
            inChannels: 64,
            outChannels: 256,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            useBias: true);

        // Act
        var report = TPModelAnalyzer.Analyze(conv, maxWorldSize: 4);

        // Assert
        Assert.NotNull(report);
        Assert.True(report.Layers.Count >= 2); // weight and bias
    }

    [Fact]
    public void Analyze_LargeConv2dOutputChannels_SuggestsParallelism()
    {
        // Arrange - Create a Conv2d with many output channels
        var conv = new Conv2d(
            inChannels: 64,
            outChannels: 256,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            useBias: false);

        // Act
        var report = TPModelAnalyzer.Analyze(conv, maxWorldSize: 4);

        // Assert
        var weightLayer = report.Layers.Find(l => l.LayerName.EndsWith(".weight"));
        Assert.NotNull(weightLayer);
        // Conv2d with >64 output channels should be parallelizable
        Assert.True(weightLayer.IsParallelizable);
    }

    [Fact]
    public void Analyze_MemoryCalculation_IsAccurate()
    {
        // Arrange
        var linear = new Linear(inFeatures: 128, outFeatures: 256, useBias: true);

        // Calculate expected memory
        // Weight: 256 * 128 * 4 bytes = 131,072 bytes
        // Bias: 256 * 4 bytes = 1,024 bytes
        long expectedWeightMemory = 256 * 128 * 4;
        long expectedBiasMemory = 256 * 4;
        long expectedTotal = expectedWeightMemory + expectedBiasMemory;

        // Act
        var report = TPModelAnalyzer.Analyze(linear, maxWorldSize: 4);

        // Assert
        Assert.Equal(expectedTotal, report.TotalMemoryBytes);
    }

    [Fact]
    public void Analyze_WorldSizeSuggestion_IsReasonable()
    {
        // Arrange - Create a large model
        var linear = new Linear(inFeatures: 4096, outFeatures: 8192, useBias: false);

        // Act
        var report = TPModelAnalyzer.Analyze(linear, maxWorldSize: 8);

        // Assert
        Assert.True(report.SuggestedWorldSize >= 1);
        Assert.True(report.SuggestedWorldSize <= 8);
    }

    [Fact]
    public void Analyze_NoParallelizableMemory_SuggestsWorldSize1()
    {
        // Arrange - Create a small non-parallelizable layer
        var linear = new Linear(inFeatures: 32, outFeatures: 32, useBias: false);

        // Act
        var report = TPModelAnalyzer.Analyze(linear, maxWorldSize: 8);

        // Assert
        Assert.Equal(1, report.SuggestedWorldSize);
    }

    [Fact]
    public void Analyze_ParallelizablePercentage_CalculatesCorrectly()
    {
        // Arrange
        var linear = new Linear(inFeatures: 128, outFeatures: 512, useBias: true);

        // Act
        var report = TPModelAnalyzer.Analyze(linear, maxWorldSize: 4);

        // Assert
        Assert.True(report.ParallelizablePercentage >= 0);
        Assert.True(report.ParallelizablePercentage <= 100);
    }

    [Fact]
    public void Analyze_GenerateSummary_ProducesValidOutput()
    {
        // Arrange
        var linear = new Linear(inFeatures: 256, outFeatures: 512, useBias: true);
        var report = TPModelAnalyzer.Analyze(linear, maxWorldSize: 4);

        // Act
        var summary = report.GenerateSummary();

        // Assert
        Assert.NotNull(summary);
        Assert.Contains("TP Model Analysis", summary);
        Assert.Contains("Total layers:", summary);
        Assert.Contains("Total memory:", summary);
        Assert.Contains("Recommendations:", summary);
    }

    [Fact]
    public void Analyze_HighParallelizablePercentage_GeneratesCorrectRecommendation()
    {
        // Arrange - Create a model with large parallelizable layers
        var linear1 = new Linear(inFeatures: 1024, outFeatures: 4096, useBias: false);
        var linear2 = new Linear(inFeatures: 4096, outFeatures: 1024, useBias: false);

        // Wrap in a hierarchical module (simulated)
        // For now, just test one layer
        var report = TPModelAnalyzer.Analyze(linear1, maxWorldSize: 4);

        // Act & Assert
        if (report.ParallelizablePercentage > 80)
        {
            Assert.Contains("Highly recommended", report.Recommendations.Values);
        }
    }
}
