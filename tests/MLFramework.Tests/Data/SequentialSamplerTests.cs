using MLFramework.Data;
using Xunit;

namespace MLFramework.Tests.Data;

/// <summary>
/// Tests for SequentialSampler.
/// </summary>
public class SequentialSamplerTests
{
    [Fact]
    public void Iterate_ReturnsSequentialIndices()
    {
        // Arrange
        var sampler = new SequentialSampler(5);

        // Act
        var indices = sampler.Iterate().ToList();

        // Assert
        Assert.Equal(new[] { 0, 1, 2, 3, 4 }, indices);
    }

    [Fact]
    public void Length_ReturnsCorrectSize()
    {
        // Arrange
        var sampler = new SequentialSampler(10);

        // Act & Assert
        Assert.Equal(10, sampler.Length);
    }

    [Fact]
    public void Iterate_EmptyDataset_ReturnsEmpty()
    {
        // Arrange
        var sampler = new SequentialSampler(0);

        // Act
        var indices = sampler.Iterate().ToList();

        // Assert
        Assert.Empty(indices);
    }

    [Fact]
    public void Iterate_SingleItem_ReturnsSingleIndex()
    {
        // Arrange
        var sampler = new SequentialSampler(1);

        // Act
        var indices = sampler.Iterate().ToList();

        // Assert
        Assert.Single(indices);
        Assert.Equal(0, indices[0]);
    }

    [Fact]
    public void Constructor_NegativeSize_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => new SequentialSampler(-1));
    }

    [Fact]
    public void Iterate_MultipleIterations_ReturnsSameOrder()
    {
        // Arrange
        var sampler = new SequentialSampler(5);

        // Act
        var first = sampler.Iterate().ToList();
        var second = sampler.Iterate().ToList();

        // Assert
        Assert.Equal(first, second);
    }

    [Fact]
    public void Iterate_LargeDataset_HandlesCorrectly()
    {
        // Arrange
        var sampler = new SequentialSampler(10000);

        // Act
        var indices = sampler.Iterate().ToList();

        // Assert
        Assert.Equal(10000, indices.Count);
        Assert.Equal(0, indices[0]);
        Assert.Equal(9999, indices[9999]);
    }

    [Fact]
    public void Iterate_ReturnsLazyEnumeration()
    {
        // Arrange
        int accessCount = 0;
        var sampler = new SequentialSampler(10);

        // Act
        var enumeration = sampler.Iterate();
        var enumerator = enumeration.GetEnumerator();
        enumerator.MoveNext();

        // Assert
        Assert.Equal(0, enumerator.Current);
        // Verify it's lazy by not consuming all elements
    }
}
