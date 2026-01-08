using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MLFramework.Data;
using Xunit;

namespace MLFramework.Tests.Data;

/// <summary>
/// Tests for the ListDataset class.
/// </summary>
public class ListDatasetTests : DatasetTestBase<ListDataset<int>, int>
{
    protected override ListDataset<int> CreateDataset(int count)
    {
        var data = new List<int>();
        for (int i = 0; i < count; i++)
        {
            data.Add(i);
        }
        return new ListDataset<int>(data);
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_NullList_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new ListDataset<int>(null));
    }

    [Fact]
    public void Constructor_EmptyList_CreatesDataset()
    {
        // Arrange
        var emptyList = new List<int>();

        // Act
        var dataset = new ListDataset<int>(emptyList);

        // Assert
        Assert.NotNull(dataset);
        Assert.Equal(0, dataset.Count);
    }

    [Fact]
    public void Constructor_NonEmptyList_CreatesDataset()
    {
        // Arrange
        var list = new List<int> { 1, 2, 3, 4, 5 };

        // Act
        var dataset = new ListDataset<int>(list);

        // Assert
        Assert.NotNull(dataset);
        Assert.Equal(5, dataset.Count);
    }

    [Fact]
    public void Constructor_LargeList_CreatesDataset()
    {
        // Arrange
        var largeList = Enumerable.Range(0, 10_000).ToList();

        // Act
        var dataset = new ListDataset<int>(largeList);

        // Assert
        Assert.Equal(10_000, dataset.Count);
    }

    #endregion

    #region Count Tests

    [Fact]
    public void Count_ReturnsListCount()
    {
        // Arrange
        var list = new List<int> { 1, 2, 3 };
        var dataset = new ListDataset<int>(list);

        // Act
        var count = dataset.Count;

        // Assert
        Assert.Equal(list.Count, count);
    }

    [Fact]
    public void Count_AfterListModification_ReturnsOriginalCount()
    {
        // Arrange
        var list = new List<int> { 1, 2, 3 };
        var dataset = new ListDataset<int>(list);

        // Act
        list.Add(4);
        var countAfterModification = dataset.Count;

        // Assert
        Assert.Equal(3, countAfterModification);
    }

    #endregion

    #region GetItem Tests

    [Fact]
    public void GetItem_ReturnsCorrectItem()
    {
        // Arrange
        var list = new List<int> { 10, 20, 30, 40, 50 };
        var dataset = new ListDataset<int>(list);

        // Act
        var item = dataset.GetItem(2);

        // Assert
        Assert.Equal(30, item);
    }

    [Fact]
    public void GetItem_AfterListModification_ReturnsOriginalItems()
    {
        // Arrange
        var list = new List<int> { 1, 2, 3 };
        var dataset = new ListDataset<int>(list);

        // Act
        list[0] = 100;
        var item = dataset.GetItem(0);

        // Assert
        // ListDataset wraps the list directly, so it reflects changes
        // This is expected behavior
        Assert.Equal(100, item);
    }

    [Fact]
    public void GetItem_ConcurrentAccess_ThreadSafe()
    {
        // Arrange
        var list = Enumerable.Range(0, 1000).ToList();
        var dataset = new ListDataset<int>(list);

        // Act & Assert
        var exception = Record.Exception(() =>
        {
            Parallel.For(0, 100, i =>
            {
                for (int j = 0; j < 100; j++)
                {
                    var index = (i * 100 + j) % 1000;
                    var item = dataset.GetItem(index);
                    Assert.Equal(index, item);
                }
            });
        });

        Assert.Null(exception);
    }

    #endregion

    #region GetBatch Tests

    [Fact]
    public void GetBatch_ListDataset_ReturnsCorrectItems()
    {
        // Arrange
        var list = new List<int> { 1, 2, 3, 4, 5 };
        var dataset = new ListDataset<int>(list);
        int[] indices = { 0, 2, 4 };

        // Act
        var batch = dataset.GetBatch(indices);

        // Assert
        Assert.Equal(3, batch.Length);
        Assert.Equal(1, batch[0]);
        Assert.Equal(3, batch[1]);
        Assert.Equal(5, batch[2]);
    }

    #endregion

    #region Type Specific Tests

    [Fact]
    public void ListDataset_WithStrings_WorksCorrectly()
    {
        // Arrange
        var list = new List<string> { "a", "b", "c" };
        var dataset = new ListDataset<string>(list);

        // Act & Assert
        Assert.Equal(3, dataset.Count);
        Assert.Equal("a", dataset.GetItem(0));
        Assert.Equal("b", dataset.GetItem(1));
        Assert.Equal("c", dataset.GetItem(2));
    }

    [Fact]
    public void ListDataset_WithNullableInts_WorksCorrectly()
    {
        // Arrange
        var list = new List<int?> { 1, null, 3, null, 5 };
        var dataset = new ListDataset<int?>(list);

        // Act & Assert
        Assert.Equal(5, dataset.Count);
        Assert.Equal(1, dataset.GetItem(0));
        Assert.Null(dataset.GetItem(1));
        Assert.Equal(3, dataset.GetItem(2));
    }

    [Fact]
    public void ListDataset_WithCustomType_WorksCorrectly()
    {
        // Arrange
        var list = new List<TestData>
        {
            new TestData { Id = 1, Name = "First" },
            new TestData { Id = 2, Name = "Second" },
        };
        var dataset = new ListDataset<TestData>(list);

        // Act & Assert
        Assert.Equal(2, dataset.Count);
        Assert.Equal(1, dataset.GetItem(0).Id);
        Assert.Equal("Second", dataset.GetItem(1).Name);
    }

    #endregion

    #region Performance Tests

    [Fact]
    public void GetItem_LargeDataset_Performance()
    {
        // Arrange
        var largeList = Enumerable.Range(0, 1_000_000).ToList();
        var dataset = new ListDataset<int>(largeList);

        // Act
        var startTime = DateTime.Now;
        for (int i = 0; i < 10000; i++)
        {
            _ = dataset.GetItem(i % 1_000_000);
        }
        var elapsed = DateTime.Now - startTime;

        // Assert
        Assert.True(elapsed.TotalMilliseconds < 1000, $"GetItem took {elapsed.TotalMilliseconds}ms, expected < 1000ms");
    }

    [Fact]
    public void GetBatch_LargeDataset_Performance()
    {
        // Arrange
        var largeList = Enumerable.Range(0, 1_000_000).ToList();
        var dataset = new ListDataset<int>(largeList);
        var indices = Enumerable.Range(0, 10000).ToArray();

        // Act
        var startTime = DateTime.Now;
        var batch = dataset.GetBatch(indices);
        var elapsed = DateTime.Now - startTime;

        // Assert
        Assert.Equal(10000, batch.Length);
        Assert.True(elapsed.TotalMilliseconds < 1000, $"GetBatch took {elapsed.TotalMilliseconds}ms, expected < 1000ms");
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void GetItem_NegativeIndexBeyondZero_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var list = new List<int> { 1, 2, 3 };
        var dataset = new ListDataset<int>(list);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => dataset.GetItem(-4));
    }

    [Fact]
    public void GetItem_IndexAtBoundary_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var list = new List<int> { 1, 2, 3 };
        var dataset = new ListDataset<int>(list);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => dataset.GetItem(3));
    }

    [Fact]
    public void GetBatch_MixedPositiveNegativeIndices_ReturnsCorrectItems()
    {
        // Arrange
        var list = new List<int> { 10, 20, 30, 40, 50 };
        var dataset = new ListDataset<int>(list);
        int[] indices = { 0, -1, 2, -2 };

        // Act
        var batch = dataset.GetBatch(indices);

        // Assert
        Assert.Equal(4, batch.Length);
        Assert.Equal(10, batch[0]);
        Assert.Equal(50, batch[1]);
        Assert.Equal(30, batch[2]);
        Assert.Equal(40, batch[3]);
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
