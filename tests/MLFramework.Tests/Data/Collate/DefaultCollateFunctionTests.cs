using MLFramework.Data.Collate;
using Xunit;

namespace MLFramework.Tests.Data.Collate;

/// <summary>
/// Tests for DefaultCollateFunction.
/// </summary>
public class DefaultCollateFunctionTests
{
    [Fact]
    public void Collate_ReturnsInputArrayUnchanged()
    {
        // Arrange
        var collator = new DefaultCollateFunction<int>();
        var batch = new[] { 1, 2, 3, 4, 5 };

        // Act
        var result = collator.Collate(batch);

        // Assert
        Assert.Same(batch, result);
    }

    [Fact]
    public void Collate_NullBatch_ThrowsArgumentNullException()
    {
        // Arrange
        var collator = new DefaultCollateFunction<int>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => collator.Collate(null));
    }

    [Fact]
    public void Collate_EmptyArray_ReturnsEmptyArray()
    {
        // Arrange
        var collator = new DefaultCollateFunction<int>();
        var batch = Array.Empty<int>();

        // Act
        var result = collator.Collate(batch) as int[];

        // Assert
        Assert.NotNull(result);
        Assert.Empty(result);
    }

    [Fact]
    public void Collate_SingleItem_ReturnsSingleItemArray()
    {
        // Arrange
        var collator = new DefaultCollateFunction<int>();
        var batch = new[] { 42 };

        // Act
        var result = collator.Collate(batch) as int[];

        // Assert
        Assert.NotNull(result);
        Assert.Single(result);
        Assert.Equal(42, result[0]);
    }

    [Fact]
    public void Collate_FloatArray_ReturnsInputArray()
    {
        // Arrange
        var collator = new DefaultCollateFunction<float>();
        var batch = new[] { 1.0f, 2.5f, 3.7f };

        // Act
        var result = collator.Collate(batch) as float[];

        // Assert
        Assert.Same(batch, result);
        Assert.Equal(new[] { 1.0f, 2.5f, 3.7f }, result);
    }

    [Fact]
    public void Collate_StringArray_ReturnsInputArray()
    {
        // Arrange
        var collator = new DefaultCollateFunction<string>();
        var batch = new[] { "a", "b", "c" };

        // Act
        var result = collator.Collate(batch) as string[];

        // Assert
        Assert.Same(batch, result);
    }

    [Fact]
    public void Collate_LargeBatch_ReturnsInputArray()
    {
        // Arrange
        var collator = new DefaultCollateFunction<int>();
        var batch = new int[10000];
        for (int i = 0; i < batch.Length; i++)
        {
            batch[i] = i;
        }

        // Act
        var result = collator.Collate(batch) as int[];

        // Assert
        Assert.Same(batch, result);
        Assert.Equal(10000, result.Length);
    }
}
