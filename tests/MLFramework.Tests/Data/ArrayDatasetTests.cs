using System;
using System.Linq;
using System.Threading.Tasks;
using MLFramework.Data;
using Xunit;

namespace MLFramework.Tests.Data;

/// <summary>
/// Tests for the ArrayDataset class.
/// </summary>
public class ArrayDatasetTests : DatasetTestBase<ArrayDataset<int>, int>
{
    protected override ArrayDataset<int> CreateDataset(int count)
    {
        var data = new int[count];
        for (int i = 0; i < count; i++)
        {
            data[i] = i;
        }
        return new ArrayDataset<int>(data);
    }

    protected override ArrayDataset<int> CreateDataset(int[] items)
    {
        return new ArrayDataset<int>(items);
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_NullArray_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new ArrayDataset<int>(null));
    }

    [Fact]
    public void Constructor_EmptyArray_CreatesDataset()
    {
        // Arrange
        var emptyArray = Array.Empty<int>();

        // Act
        var dataset = new ArrayDataset<int>(emptyArray);

        // Assert
        Assert.NotNull(dataset);
        Assert.Equal(0, dataset.Count);
    }

    [Fact]
    public void Constructor_NonEmptyArray_CreatesDataset()
    {
        // Arrange
        var array = new int[] { 1, 2, 3, 4, 5 };

        // Act
        var dataset = new ArrayDataset<int>(array);

        // Assert
        Assert.NotNull(dataset);
        Assert.Equal(5, dataset.Count);
    }

    [Fact]
    public void Constructor_LargeArray_CreatesDataset()
    {
        // Arrange
        var largeArray = Enumerable.Range(0, 10_000).ToArray();

        // Act
        var dataset = new ArrayDataset<int>(largeArray);

        // Assert
        Assert.Equal(10_000, dataset.Count);
    }

    #endregion

    #region Count Tests

    [Fact]
    public void Count_ReturnsArrayLength()
    {
        // Arrange
        var array = new int[] { 1, 2, 3 };
        var dataset = new ArrayDataset<int>(array);

        // Act
        var count = dataset.Count;

        // Assert
        Assert.Equal(array.Length, count);
    }

    [Fact]
    public void Count_AfterArrayModification_ReturnsOriginalCount()
    {
        // Arrange
        var array = new int[] { 1, 2, 3 };
        var dataset = new ArrayDataset<int>(array);

        // Act
        array[0] = 100;
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
        var array = new int[] { 10, 20, 30, 40, 50 };
        var dataset = new ArrayDataset<int>(array);

        // Act
        var item = dataset.GetItem(2);

        // Assert
        Assert.Equal(30, item);
    }

    [Fact]
    public void GetItem_DirectArrayAccess_Performance()
    {
        // Arrange
        var array = Enumerable.Range(0, 1_000_000).ToArray();
        var dataset = new ArrayDataset<int>(array);

        // Act - Time direct array access
        var startDirect = DateTime.Now;
        for (int i = 0; i < 10000; i++)
        {
            _ = array[i % 1_000_000];
        }
        var elapsedDirect = DateTime.Now - startDirect;

        // Act - Time dataset access
        var startDataset = DateTime.Now;
        for (int i = 0; i < 10000; i++)
        {
            _ = dataset.GetItem(i % 1_000_000);
        }
        var elapsedDataset = DateTime.Now - startDataset;

        // Assert - Dataset access should be reasonably close to direct access
        // Allow up to 5x slower due to method call overhead
        Assert.True(elapsedDataset.TotalMilliseconds < elapsedDirect.TotalMilliseconds * 5,
            $"Dataset access took {elapsedDataset.TotalMilliseconds}ms, direct access took {elapsedDirect.TotalMilliseconds}ms");
    }

    [Fact]
    public void GetItem_ConcurrentAccess_ThreadSafe()
    {
        // Arrange
        var array = Enumerable.Range(0, 1000).ToArray();
        var dataset = new ArrayDataset<int>(array);

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
    public void GetBatch_ArrayDataset_ReturnsCorrectItems()
    {
        // Arrange
        var array = new int[] { 1, 2, 3, 4, 5 };
        var dataset = new ArrayDataset<int>(array);
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
    public void ArrayDataset_WithStrings_WorksCorrectly()
    {
        // Arrange
        var array = new string[] { "a", "b", "c" };
        var dataset = new ArrayDataset<string>(array);

        // Act & Assert
        Assert.Equal(3, dataset.Count);
        Assert.Equal("a", dataset.GetItem(0));
        Assert.Equal("b", dataset.GetItem(1));
        Assert.Equal("c", dataset.GetItem(2));
    }

    [Fact]
    public void ArrayDataset_WithNullableInts_WorksCorrectly()
    {
        // Arrange
        var array = new int?[] { 1, null, 3, null, 5 };
        var dataset = new ArrayDataset<int?>(array);

        // Act & Assert
        Assert.Equal(5, dataset.Count);
        Assert.Equal(1, dataset.GetItem(0));
        Assert.Null(dataset.GetItem(1));
        Assert.Equal(3, dataset.GetItem(2));
    }

    [Fact]
    public void ArrayDataset_WithDoubles_WorksCorrectly()
    {
        // Arrange
        var array = new double[] { 1.1, 2.2, 3.3 };
        var dataset = new ArrayDataset<double>(array);

        // Act & Assert
        Assert.Equal(3, dataset.Count);
        Assert.Equal(1.1, dataset.GetItem(0));
        Assert.Equal(2.2, dataset.GetItem(1));
        Assert.Equal(3.3, dataset.GetItem(2));
    }

    [Fact]
    public void ArrayDataset_WithCustomType_WorksCorrectly()
    {
        // Arrange
        var array = new TestData[]
        {
            new TestData { Id = 1, Name = "First" },
            new TestData { Id = 2, Name = "Second" },
        };
        var dataset = new ArrayDataset<TestData>(array);

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
        var largeArray = Enumerable.Range(0, 1_000_000).ToArray();
        var dataset = new ArrayDataset<int>(largeArray);

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
        var largeArray = Enumerable.Range(0, 1_000_000).ToArray();
        var dataset = new ArrayDataset<int>(largeArray);
        var indices = Enumerable.Range(0, 10000).ToArray();

        // Act
        var startTime = DateTime.Now;
        var batch = dataset.GetBatch(indices);
        var elapsed = DateTime.Now - startTime;

        // Assert
        Assert.Equal(10000, batch.Length);
        Assert.True(elapsed.TotalMilliseconds < 1000, $"GetBatch took {elapsed.TotalMilliseconds}ms, expected < 1000ms");
    }

    [Fact]
    public void GetItem_ArrayDataset_VsListDataset_Performance()
    {
        // Arrange
        var array = Enumerable.Range(0, 100_000).ToArray();
        var list = Enumerable.Range(0, 100_000).ToList();
        var arrayDataset = new ArrayDataset<int>(array);
        var listDataset = new ListDataset<int>(list);

        // Act - Time array dataset
        var startArray = DateTime.Now;
        for (int i = 0; i < 100000; i++)
        {
            _ = arrayDataset.GetItem(i % 100_000);
        }
        var elapsedArray = DateTime.Now - startArray;

        // Act - Time list dataset
        var startList = DateTime.Now;
        for (int i = 0; i < 100000; i++)
        {
            _ = listDataset.GetItem(i % 100_000);
        }
        var elapsedList = DateTime.Now - startList;

        // Assert - Array dataset should be faster or at least comparable
        // Allow list to be up to 2x slower
        Assert.True(elapsedArray.TotalMilliseconds < elapsedList.TotalMilliseconds * 2,
            $"Array dataset took {elapsedArray.TotalMilliseconds}ms, list dataset took {elapsedList.TotalMilliseconds}ms");
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void GetItem_NegativeIndexBeyondZero_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var array = new int[] { 1, 2, 3 };
        var dataset = new ArrayDataset<int>(array);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => dataset.GetItem(-4));
    }

    [Fact]
    public void GetItem_IndexAtBoundary_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var array = new int[] { 1, 2, 3 };
        var dataset = new ArrayDataset<int>(array);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => dataset.GetItem(3));
    }

    [Fact]
    public void GetBatch_MixedPositiveNegativeIndices_ReturnsCorrectItems()
    {
        // Arrange
        var array = new int[] { 10, 20, 30, 40, 50 };
        var dataset = new ArrayDataset<int>(array);
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

    [Fact]
    public void GetItem_ZeroLengthArray_Throws()
    {
        // Arrange
        var array = Array.Empty<int>();
        var dataset = new ArrayDataset<int>(array);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => dataset.GetItem(0));
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
