using System;
using System.Collections.Generic;
using System.Linq;
using MLFramework.Data;
using Xunit;

namespace MLFramework.Tests.Data;

/// <summary>
/// Tests for the InMemoryDataset static class.
/// </summary>
public class InMemoryDatasetTests
{
    #region FromEnumerable Tests

    [Fact]
    public void FromEnumerable_NullEnumerable_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => InMemoryDataset.FromEnumerable<int>(null));
    }

    [Fact]
    public void FromEnumerable_List_ReturnsListDataset()
    {
        // Arrange
        var list = new List<int> { 1, 2, 3, 4, 5 };

        // Act
        var dataset = InMemoryDataset.FromEnumerable(list);

        // Assert
        Assert.IsType<ListDataset<int>>(dataset);
        Assert.Equal(5, dataset.Count);
        Assert.Equal(1, dataset.GetItem(0));
        Assert.Equal(5, dataset.GetItem(4));
    }

    [Fact]
    public void FromEnumerable_IList_ReturnsListDataset()
    {
        // Arrange
        IList<int> list = new List<int> { 1, 2, 3 };

        // Act
        var dataset = InMemoryDataset.FromEnumerable(list);

        // Assert
        Assert.IsType<ListDataset<int>>(dataset);
    }

    [Fact]
    public void FromEnumerable_Array_ReturnsArrayDataset()
    {
        // Arrange
        var array = new int[] { 1, 2, 3, 4, 5 };

        // Act
        var dataset = InMemoryDataset.FromEnumerable(array);

        // Assert
        Assert.IsType<ArrayDataset<int>>(dataset);
        Assert.Equal(5, dataset.Count);
        Assert.Equal(1, dataset.GetItem(0));
        Assert.Equal(5, dataset.GetItem(4));
    }

    [Fact]
    public void FromEnumerable_GenericEnumerable_ReturnsArrayDataset()
    {
        // Arrange
        var enumerable = Enumerable.Range(0, 100).Where(x => x % 2 == 0);

        // Act
        var dataset = InMemoryDataset.FromEnumerable(enumerable);

        // Assert
        Assert.IsType<ArrayDataset<int>>(dataset);
        Assert.Equal(50, dataset.Count);
        Assert.Equal(0, dataset.GetItem(0));
        Assert.Equal(98, dataset.GetItem(49));
    }

    [Fact]
    public void FromEnumerable_LargeEnumerable_ReturnsCorrectDataset()
    {
        // Arrange
        var largeEnumerable = Enumerable.Range(0, 10_000);

        // Act
        var dataset = InMemoryDataset.FromEnumerable(largeEnumerable);

        // Assert
        Assert.Equal(10_000, dataset.Count);
        for (int i = 0; i < 10_000; i++)
        {
            Assert.Equal(i, dataset.GetItem(i));
        }
    }

    [Fact]
    public void FromEnumerable_EmptyEnumerable_ReturnsEmptyDataset()
    {
        // Arrange
        var emptyEnumerable = Enumerable.Empty<int>();

        // Act
        var dataset = InMemoryDataset.FromEnumerable(emptyEnumerable);

        // Assert
        Assert.Equal(0, dataset.Count);
    }

    [Fact]
    public void FromEnumerable_YieldReturn_ReturnsArrayDataset()
    {
        // Arrange
        IEnumerable<int> GenerateNumbers()
        {
            for (int i = 0; i < 10; i++)
            {
                yield return i;
            }
        }

        // Act
        var dataset = InMemoryDataset.FromEnumerable(GenerateNumbers());

        // Assert
        Assert.IsType<ArrayDataset<int>>(dataset);
        Assert.Equal(10, dataset.Count);
    }

    [Fact]
    public void FromEnumerable_HashSet_ReturnsArrayDataset()
    {
        // Arrange
        var hashSet = new HashSet<int> { 1, 2, 3, 4, 5 };

        // Act
        var dataset = InMemoryDataset.FromEnumerable(hashSet);

        // Assert
        Assert.IsType<ArrayDataset<int>>(dataset);
        Assert.Equal(5, dataset.Count);
        Assert.Contains(dataset.GetItem(0), hashSet);
    }

    #endregion

    #region FromList Tests

    [Fact]
    public void FromList_ReturnsListDataset()
    {
        // Arrange
        var list = new List<int> { 1, 2, 3, 4, 5 };

        // Act
        var dataset = InMemoryDataset.FromList(list);

        // Assert
        Assert.IsType<ListDataset<int>>(dataset);
        Assert.Equal(5, dataset.Count);
    }

    [Fact]
    public void FromList_IList_ReturnsListDataset()
    {
        // Arrange
        IList<int> list = new List<int> { 1, 2, 3 };

        // Act
        var dataset = InMemoryDataset.FromList(list);

        // Assert
        Assert.IsType<ListDataset<int>>(dataset);
    }

    [Fact]
    public void FromList_EmptyList_CreatesDataset()
    {
        // Arrange
        var emptyList = new List<int>();

        // Act
        var dataset = InMemoryDataset.FromList(emptyList);

        // Assert
        Assert.NotNull(dataset);
        Assert.Equal(0, dataset.Count);
    }

    [Fact]
    public void FromList_NullList_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => InMemoryDataset.FromList<int>(null));
    }

    #endregion

    #region FromArray Tests

    [Fact]
    public void FromArray_ReturnsArrayDataset()
    {
        // Arrange
        var array = new int[] { 1, 2, 3, 4, 5 };

        // Act
        var dataset = InMemoryDataset.FromArray(array);

        // Assert
        Assert.IsType<ArrayDataset<int>>(dataset);
        Assert.Equal(5, dataset.Count);
    }

    [Fact]
    public void FromArray_EmptyArray_CreatesDataset()
    {
        // Arrange
        var emptyArray = Array.Empty<int>();

        // Act
        var dataset = InMemoryDataset.FromArray(emptyArray);

        // Assert
        Assert.NotNull(dataset);
        Assert.Equal(0, dataset.Count);
    }

    [Fact]
    public void FromArray_NullArray_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => InMemoryDataset.FromArray<int>(null));
    }

    #endregion

    #region Type Specific Tests

    [Fact]
    public void FromEnumerable_WithStrings_WorksCorrectly()
    {
        // Arrange
        var enumerable = new List<string> { "a", "b", "c" };

        // Act
        var dataset = InMemoryDataset.FromEnumerable(enumerable);

        // Assert
        Assert.Equal(3, dataset.Count);
        Assert.Equal("a", dataset.GetItem(0));
        Assert.Equal("b", dataset.GetItem(1));
        Assert.Equal("c", dataset.GetItem(2));
    }

    [Fact]
    public void FromEnumerable_WithDoubles_WorksCorrectly()
    {
        // Arrange
        var enumerable = new double[] { 1.1, 2.2, 3.3 };

        // Act
        var dataset = InMemoryDataset.FromEnumerable(enumerable);

        // Assert
        Assert.Equal(3, dataset.Count);
        Assert.Equal(1.1, dataset.GetItem(0));
        Assert.Equal(2.2, dataset.GetItem(1));
        Assert.Equal(3.3, dataset.GetItem(2));
    }

    [Fact]
    public void FromEnumerable_WithNullableInts_WorksCorrectly()
    {
        // Arrange
        var enumerable = new int?[] { 1, null, 3, null, 5 };

        // Act
        var dataset = InMemoryDataset.FromEnumerable(enumerable);

        // Assert
        Assert.Equal(5, dataset.Count);
        Assert.Equal(1, dataset.GetItem(0));
        Assert.Null(dataset.GetItem(1));
        Assert.Equal(3, dataset.GetItem(2));
    }

    [Fact]
    public void FromEnumerable_WithCustomType_WorksCorrectly()
    {
        // Arrange
        var enumerable = new List<TestData>
        {
            new TestData { Id = 1, Name = "First" },
            new TestData { Id = 2, Name = "Second" },
            new TestData { Id = 3, Name = "Third" },
        };

        // Act
        var dataset = InMemoryDataset.FromEnumerable(enumerable);

        // Assert
        Assert.Equal(3, dataset.Count);
        Assert.Equal(1, dataset.GetItem(0).Id);
        Assert.Equal("Second", dataset.GetItem(1).Name);
        Assert.Equal(3, dataset.GetItem(2).Id);
    }

    #endregion

    #region Optimization Tests

    [Fact]
    public void FromEnumerable_ArrayDoesNotCopy_ReturnsSameArray()
    {
        // Arrange
        var array = new int[] { 1, 2, 3 };

        // Act
        var dataset = InMemoryDataset.FromEnumerable(array) as ArrayDataset<int>;

        // Assert
        Assert.NotNull(dataset);
        // The dataset should wrap the array directly without copying
        // We can verify this by modifying the array and seeing if the dataset reflects it
        array[0] = 999;
        Assert.Equal(999, dataset.GetItem(0));
    }

    [Fact]
    public void FromEnumerable_ListDoesNotCopy_ReturnsSameList()
    {
        // Arrange
        var list = new List<int> { 1, 2, 3 };

        // Act
        var dataset = InMemoryDataset.FromEnumerable(list) as ListDataset<int>;

        // Assert
        Assert.NotNull(dataset);
        // The dataset should wrap the list directly without copying
        // We can verify this by modifying the list and seeing if the dataset reflects it
        list[0] = 999;
        Assert.Equal(999, dataset.GetItem(0));
    }

    [Fact]
    public void FromEnumerable_GenericEnumerable_MaterializesToNewArray()
    {
        // Arrange
        int callCount = 0;
        IEnumerable<int> GenerateNumbers()
        {
            callCount++;
            for (int i = 0; i < 10; i++)
            {
                yield return i;
            }
        }

        // Act
        var dataset = InMemoryDataset.FromEnumerable(GenerateNumbers());
        var firstCallCount = callCount;

        // Act - Access items again
        _ = dataset.GetItem(0);
        _ = dataset.GetItem(5);

        // Assert
        // The enumerable should only be enumerated once during materialization
        Assert.Equal(1, firstCallCount);
        Assert.Equal(1, callCount); // No additional calls
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void FromEnumerable_SingleItem_WorksCorrectly()
    {
        // Arrange
        var enumerable = new int[] { 42 };

        // Act
        var dataset = InMemoryDataset.FromEnumerable(enumerable);

        // Assert
        Assert.Equal(1, dataset.Count);
        Assert.Equal(42, dataset.GetItem(0));
    }

    [Fact]
    public void FromEnumerable_WithNullItems_WorksCorrectly()
    {
        // Arrange
        var enumerable = new int?[] { 1, null, null, 4 };

        // Act
        var dataset = InMemoryDataset.FromEnumerable(enumerable);

        // Assert
        Assert.Equal(4, dataset.Count);
        Assert.Equal(1, dataset.GetItem(0));
        Assert.Null(dataset.GetItem(1));
        Assert.Null(dataset.GetItem(2));
        Assert.Equal(4, dataset.GetItem(3));
    }

    [Fact]
    public void FromEnumerable_LargeRange_Performance()
    {
        // Arrange
        var largeEnumerable = Enumerable.Range(0, 100_000);

        // Act
        var startTime = DateTime.Now;
        var dataset = InMemoryDataset.FromEnumerable(largeEnumerable);
        var elapsed = DateTime.Now - startTime;

        // Assert
        Assert.Equal(100_000, dataset.Count);
        Assert.True(elapsed.TotalMilliseconds < 1000, $"FromEnumerable took {elapsed.TotalMilliseconds}ms, expected < 1000ms");
    }

    [Fact]
    public void FromList_ListDatasetPreservesReference()
    {
        // Arrange
        var list = new List<int> { 1, 2, 3 };

        // Act
        var dataset = InMemoryDataset.FromList(list);

        // Assert
        Assert.IsType<ListDataset<int>>(dataset);
        // Modify the list and verify dataset reflects the change
        list[0] = 999;
        Assert.Equal(999, dataset.GetItem(0));
    }

    [Fact]
    public void FromArray_ArrayDatasetPreservesReference()
    {
        // Arrange
        var array = new int[] { 1, 2, 3 };

        // Act
        var dataset = InMemoryDataset.FromArray(array);

        // Assert
        Assert.IsType<ArrayDataset<int>>(dataset);
        // Modify the array and verify dataset reflects the change
        array[0] = 999;
        Assert.Equal(999, dataset.GetItem(0));
    }

    #endregion

    #region Test Helper Class

    private class TestData
    {
        public int Id { get; set; }
        public string Name { get; set; }
    }

    #endregion
}
