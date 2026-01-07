using Xunit;

namespace MLFramework.Tests.Memory;

public class AdaptivePaddingStrategyTests
{
    private readonly ShapeBounds _bounds;
    private const int ElementSize = 4;

    public AdaptivePaddingStrategyTests()
    {
        _bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });
    }

    [Fact]
    public void Constructor_DefaultValues_CreatesInstance()
    {
        var strategy = new AdaptivePaddingStrategy();

        Assert.Equal(1.5, strategy.GrowthFactor);
        Assert.Equal(0.5, strategy.ShrinkThreshold);
    }

    [Fact]
    public void Constructor_CustomValues_SetsProperties()
    {
        var strategy = new AdaptivePaddingStrategy(growthFactor: 2.0, shrinkThreshold: 0.3);

        Assert.Equal(2.0, strategy.GrowthFactor);
        Assert.Equal(0.3, strategy.ShrinkThreshold);
    }

    [Fact]
    public void Constructor_InvalidGrowthFactor_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
        {
            new AdaptivePaddingStrategy(growthFactor: 1.0);
        });

        Assert.Throws<ArgumentException>(() =>
        {
            new AdaptivePaddingStrategy(growthFactor: 0.5);
        });
    }

    [Fact]
    public void Constructor_InvalidShrinkThreshold_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
        {
            new AdaptivePaddingStrategy(shrinkThreshold: 0.0);
        });

        Assert.Throws<ArgumentException>(() =>
        {
            new AdaptivePaddingStrategy(shrinkThreshold: 1.0);
        });
    }

    [Fact]
    public void CalculateRequiredSize_AppliesGrowthFactor()
    {
        var strategy = new AdaptivePaddingStrategy(growthFactor: 1.5);

        long size = strategy.CalculateRequiredSize(_bounds, ElementSize);

        // Expected: 2 * 15 = 30 elements
        // With growth: 30 * 1.5 = 45 elements
        // In bytes: 45 * 4 = 180 bytes
        Assert.Equal(180, size);
    }

    [Fact]
    public void CalculateRequiredSize_CapsAtMaxShape()
    {
        // Use a large growth factor to test capping
        var strategy = new AdaptivePaddingStrategy(growthFactor: 10.0);

        long size = strategy.CalculateRequiredSize(_bounds, ElementSize);

        // Max shape: 5 * 20 = 100 elements
        // Expected with growth would be 300, but should cap at 100
        // In bytes: 100 * 4 = 400 bytes
        Assert.Equal(400, size);
    }

    [Fact]
    public void ShouldResize_NeedsMoreSpace_ReturnsTrue()
    {
        var strategy = new AdaptivePaddingStrategy();
        var handle = new MemoryHandle(
            new IntPtr(1000),
            120L, // Enough for 2*15*4
            new[] { 2, 15 },
            _bounds,
            ElementSize);

        bool shouldResize = strategy.ShouldResize(handle, new[] { 5, 20 });

        Assert.True(shouldResize);
    }

    [Fact]
    public void ShouldResize_LowUtilization_ReturnsTrue()
    {
        var strategy = new AdaptivePaddingStrategy();
        var handle = new MemoryHandle(
            new IntPtr(1000),
            1000L, // Large capacity
            new[] { 2, 15 },
            _bounds,
            ElementSize);

        bool shouldResize = strategy.ShouldResize(handle, new[] { 1, 10 });

        Assert.True(shouldResize);
    }

    [Fact]
    public void ShouldResize_SufficientCapacity_ReturnsFalse()
    {
        var strategy = new AdaptivePaddingStrategy();
        var handle = new MemoryHandle(
            new IntPtr(1000),
            200L, // Enough for growth
            new[] { 2, 15 },
            _bounds,
            ElementSize);

        bool shouldResize = strategy.ShouldResize(handle, new[] { 3, 15 });

        Assert.False(shouldResize);
    }

    [Fact]
    public void ShouldResize_ModerateUtilization_ReturnsFalse()
    {
        var strategy = new AdaptivePaddingStrategy();
        var handle = new MemoryHandle(
            new IntPtr(1000),
            200L,
            new[] { 2, 15 },
            _bounds,
            ElementSize);

        bool shouldResize = strategy.ShouldResize(handle, new[] { 2, 12 });

        // 2 * 12 * 4 = 96 bytes, utilization = 96/200 = 0.48 (close to threshold)
        // Should not shrink because it's not significantly below expected
        Assert.False(shouldResize);
    }
}
