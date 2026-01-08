using Xunit;

namespace MLFramework.Tests.Memory;

public class ShapeBoundsTests
{
    [Fact]
    public void Constructor_ValidBounds_CreatesInstance()
    {
        var minShape = new[] { 1, 10 };
        var maxShape = new[] { 5, 20 };
        var expectedShape = new[] { 2, 15 };

        var bounds = new ShapeBounds(minShape, maxShape, expectedShape);

        Assert.Equal(minShape, bounds.MinShape);
        Assert.Equal(maxShape, bounds.MaxShape);
        Assert.Equal(expectedShape, bounds.ExpectedShape);
    }

    [Fact]
    public void Constructor_DifferentLengths_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
        {
            new ShapeBounds(new[] { 1, 2 }, new[] { 3, 4, 5 }, new[] { 2, 3 });
        });
    }

    [Fact]
    public void Constructor_NegativeMinShape_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
        {
            new ShapeBounds(new[] { -1, 2 }, new[] { 3, 4 }, new[] { 2, 3 });
        });
    }

    [Fact]
    public void Constructor_MaxLessThanMin_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
        {
            new ShapeBounds(new[] { 5, 2 }, new[] { 3, 4 }, new[] { 4, 3 });
        });
    }

    [Fact]
    public void Constructor_ExpectedOutOfBounds_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
        {
            new ShapeBounds(new[] { 1, 2 }, new[] { 3, 4 }, new[] { 4, 3 });
        });
    }

    [Fact]
    public void CalculateMaxElements_ReturnsCorrectValue()
    {
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });

        long maxElements = bounds.CalculateMaxElements();

        Assert.Equal(100, maxElements); // 5 * 20
    }

    [Fact]
    public void CalculateExpectedElements_ReturnsCorrectValue()
    {
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });

        long expectedElements = bounds.CalculateExpectedElements();

        Assert.Equal(30, expectedElements); // 2 * 15
    }

    [Fact]
    public void CalculateElements_ReturnsCorrectValue()
    {
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });

        long elements = bounds.CalculateElements(new[] { 3, 12 });

        Assert.Equal(36, elements); // 3 * 12
    }

    [Fact]
    public void CalculateElements_WrongRank_ThrowsArgumentException()
    {
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });

        Assert.Throws<ArgumentException>(() =>
        {
            bounds.CalculateElements(new[] { 1, 2, 3 });
        });
    }

    [Fact]
    public void Contains_ValidShape_ReturnsTrue()
    {
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });

        bool result = bounds.Contains(new[] { 3, 15 });

        Assert.True(result);
    }

    [Fact]
    public void Contains_ShapeBelowMin_ReturnsFalse()
    {
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });

        bool result = bounds.Contains(new[] { 0, 15 });

        Assert.False(result);
    }

    [Fact]
    public void Contains_ShapeAboveMax_ReturnsFalse()
    {
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });

        bool result = bounds.Contains(new[] { 6, 15 });

        Assert.False(result);
    }

    [Fact]
    public void Contains_WrongRank_ReturnsFalse()
    {
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });

        bool result = bounds.Contains(new[] { 1, 2, 3 });

        Assert.False(result);
    }

    [Fact]
    public void ToString_ReturnsFormattedString()
    {
        var bounds = new ShapeBounds(new[] { 1, 10 }, new[] { 5, 20 }, new[] { 2, 15 });

        string result = bounds.ToString();

        Assert.Contains("[1x10 -> 5x20]", result);
        Assert.Contains("(expected: 2x15)", result);
    }
}
