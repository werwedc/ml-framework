using MLFramework.Fusion.Dynamic;

namespace MLFramework.Tests.Fusion.Dynamic;

/// <summary>
/// Unit tests for FusionBenefit
/// </summary>
public class FusionBenefitTests
{
    [Fact]
    public void ShouldFuse_WithHighSpeedup_ReturnsTrue()
    {
        // Arrange
        var benefit = FusionBenefit.Create(speedup: 1.5, memorySaved: 1024, kernelReduction: 2, complexity: 2.0);

        // Act
        var result = benefit.ShouldFuse(threshold: 1.1);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public void ShouldFuse_WithLowSpeedup_ReturnsFalse()
    {
        // Arrange
        var benefit = FusionBenefit.Create(speedup: 1.05, memorySaved: 1024, kernelReduction: 2, complexity: 2.0);

        // Act
        var result = benefit.ShouldFuse(threshold: 1.1);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ShouldFuse_WithNoKernelReduction_ReturnsFalse()
    {
        // Arrange
        var benefit = FusionBenefit.Create(speedup: 1.5, memorySaved: 1024, kernelReduction: 0, complexity: 2.0);

        // Act
        var result = benefit.ShouldFuse(threshold: 1.1);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public void ShouldFuse_WithCustomThreshold_UsesCorrectThreshold()
    {
        // Arrange
        var benefit = FusionBenefit.Create(speedup: 1.2, memorySaved: 1024, kernelReduction: 2, complexity: 2.0);

        // Act
        var result1 = benefit.ShouldFuse(threshold: 1.1);
        var result2 = benefit.ShouldFuse(threshold: 1.3);

        // Assert
        Assert.True(result1);
        Assert.False(result2);
    }

    [Fact]
    public void None_ReturnsZeroBenefit()
    {
        // Arrange & Act
        var benefit = FusionBenefit.None();

        // Assert
        Assert.Equal(1.0, benefit.EstimatedSpeedup);
        Assert.Equal(0, benefit.MemorySaved);
        Assert.Equal(0, benefit.KernelCountReduction);
        Assert.Equal(double.MaxValue, benefit.ComplexityScore);
        Assert.False(benefit.ShouldFuse());
    }

    [Fact]
    public void Create_SetsAllPropertiesCorrectly()
    {
        // Arrange
        const double speedup = 1.8;
        const long memorySaved = 4096;
        const int kernelReduction = 3;
        const double complexity = 3.5;

        // Act
        var benefit = FusionBenefit.Create(speedup, memorySaved, kernelReduction, complexity);

        // Assert
        Assert.Equal(speedup, benefit.EstimatedSpeedup);
        Assert.Equal(memorySaved, benefit.MemorySaved);
        Assert.Equal(kernelReduction, benefit.KernelCountReduction);
        Assert.Equal(complexity, benefit.ComplexityScore);
    }
}
