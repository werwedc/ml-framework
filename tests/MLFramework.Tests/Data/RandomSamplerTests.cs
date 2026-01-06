using MLFramework.Data;
using Xunit;

namespace MLFramework.Tests.Data;

/// <summary>
/// Tests for RandomSampler.
/// </summary>
public class RandomSamplerTests
{
    [Fact]
    public void Iterate_ReturnsAllIndices()
    {
        // Arrange
        var sampler = new RandomSampler(5, seed: 42);

        // Act
        var indices = sampler.Iterate().ToList();

        // Assert
        Assert.Equal(5, indices.Count);
        Assert.Contains(0, indices);
        Assert.Contains(1, indices);
        Assert.Contains(2, indices);
        Assert.Contains(3, indices);
        Assert.Contains(4, indices);
    }

    [Fact]
    public void Length_ReturnsCorrectSize()
    {
        // Arrange
        var sampler = new RandomSampler(10, seed: 42);

        // Act & Assert
        Assert.Equal(10, sampler.Length);
    }

    [Fact]
    public void Iterate_EmptyDataset_ReturnsEmpty()
    {
        // Arrange
        var sampler = new RandomSampler(0, seed: 42);

        // Act
        var indices = sampler.Iterate().ToList();

        // Assert
        Assert.Empty(indices);
    }

    [Fact]
    public void Iterate_SingleItem_ReturnsSingleIndex()
    {
        // Arrange
        var sampler = new RandomSampler(1, seed: 42);

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
        Assert.Throws<ArgumentOutOfRangeException>(() => new RandomSampler(-1));
    }

    [Fact]
    public void Iterate_SameSeed_SameOrder()
    {
        // Arrange
        var sampler1 = new RandomSampler(10, seed: 12345);
        var sampler2 = new RandomSampler(10, seed: 12345);

        // Act
        var indices1 = sampler1.Iterate().ToList();
        var indices2 = sampler2.Iterate().ToList();

        // Assert
        Assert.Equal(indices1, indices2);
    }

    [Fact]
    public void Iterate_DifferentSeeds_DifferentOrders()
    {
        // Arrange
        var sampler1 = new RandomSampler(10, seed: 12345);
        var sampler2 = new RandomSampler(10, seed: 54321);

        // Act
        var indices1 = sampler1.Iterate().ToList();
        var indices2 = sampler2.Iterate().ToList();

        // Assert
        Assert.NotEqual(indices1, indices2);
    }

    [Fact]
    public void Iterate_NoSeed_RandomOrder()
    {
        // Arrange
        var sampler1 = new RandomSampler(10);
        var sampler2 = new RandomSampler(10);

        // Act
        var indices1 = sampler1.Iterate().ToList();
        var indices2 = sampler2.Iterate().ToList();

        // Assert - Very unlikely to be the same by chance
        Assert.NotEqual(indices1, indices2);
    }

    [Fact]
    public void Iterate_EachIndexAppearsOnce()
    {
        // Arrange
        var sampler = new RandomSampler(100, seed: 42);

        // Act
        var indices = sampler.Iterate().ToList();

        // Assert
        var distinctIndices = indices.Distinct().ToList();
        Assert.Equal(100, distinctIndices.Count);
    }

    [Fact]
    public void Iterate_LargeDataset_HandlesCorrectly()
    {
        // Arrange
        var sampler = new RandomSampler(10000, seed: 42);

        // Act
        var indices = sampler.Iterate().ToList();

        // Assert
        Assert.Equal(10000, indices.Count);
        Assert.Equal(10000, indices.Distinct().Count());
    }

    [Fact]
    public void Iterate_NotSequentialOrder()
    {
        // Arrange
        var sampler = new RandomSampler(100, seed: 42);

        // Act
        var indices = sampler.Iterate().ToList();

        // Assert - Check that it's not perfectly sequential
        bool isSequential = true;
        for (int i = 0; i < indices.Count - 1; i++)
        {
            if (indices[i + 1] != indices[i] + 1)
            {
                isSequential = false;
                break;
            }
        }
        Assert.False(isSequential);
    }
}
