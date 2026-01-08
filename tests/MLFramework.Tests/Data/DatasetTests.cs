using System;
using System.Threading.Tasks;
using MLFramework.Data;
using Xunit;

namespace MLFramework.Tests.Data;

/// <summary>
/// Abstract base class for dataset tests providing common test infrastructure.
/// </summary>
/// <typeparam name="TDataset">The type of dataset to test.</typeparam>
/// <typeparam name="T">The type of items in the dataset.</typeparam>
public abstract class DatasetTestBase<TDataset, T>
    where TDataset : IDataset<T>
{
    protected abstract TDataset CreateDataset(int count);

    protected virtual TDataset CreateDataset(T[] items)
    {
        throw new NotImplementedException();
    }

    protected static T[] CreateTestData(int count)
    {
        var data = new T[count];
        for (int i = 0; i < count; i++)
        {
            data[i] = (T)Convert.ChangeType(i, typeof(T));
        }
        return data;
    }

    protected static void AssertDatasetEquals(IDataset<T> dataset, T[] expected)
    {
        Assert.Equal(expected.Length, dataset.Count);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], dataset.GetItem(i));
        }
    }

    #region Count Property Tests

    [Fact]
    public void Count_ReturnsCorrectCount()
    {
        // Arrange
        var dataset = CreateDataset(10);

        // Act
        var count = dataset.Count;

        // Assert
        Assert.Equal(10, count);
    }

    [Fact]
    public void Count_ZeroItems_ReturnsZero()
    {
        // Arrange
        var dataset = CreateDataset(0);

        // Act
        var count = dataset.Count;

        // Assert
        Assert.Equal(0, count);
    }

    [Fact]
    public void Count_LargeDataset_ReturnsLargeNumber()
    {
        // Arrange
        var dataset = CreateDataset(1_000_000);

        // Act
        var count = dataset.Count;

        // Assert
        Assert.Equal(1_000_000, count);
    }

    #endregion

    #region GetItem Method Tests

    [Fact]
    public void GetItem_ValidIndex_ReturnsCorrectItem()
    {
        // Arrange
        var dataset = CreateDataset(10);
        int index = 5;

        // Act
        var item = dataset.GetItem(index);

        // Assert
        Assert.Equal(Convert.ChangeType(index, typeof(T)), item);
    }

    [Fact]
    public void GetItem_FirstIndex_ReturnsFirstItem()
    {
        // Arrange
        var dataset = CreateDataset(10);

        // Act
        var item = dataset.GetItem(0);

        // Assert
        Assert.Equal(default, item);
    }

    [Fact]
    public void GetItem_LastIndex_ReturnsLastItem()
    {
        // Arrange
        var dataset = CreateDataset(10);

        // Act
        var item = dataset.GetItem(9);

        // Assert
        Assert.Equal(Convert.ChangeType(9, typeof(T)), item);
    }

    [Fact]
    public void GetItem_NegativeIndex_LastItem()
    {
        // Arrange
        var dataset = CreateDataset(10);

        // Act
        var item = dataset.GetItem(-1);

        // Assert
        Assert.Equal(Convert.ChangeType(9, typeof(T)), item);
    }

    [Fact]
    public void GetItem_NegativeIndex_FirstItem()
    {
        // Arrange
        var dataset = CreateDataset(10);

        // Act
        var item = dataset.GetItem(-10);

        // Assert
        Assert.Equal(default, item);
    }

    [Fact]
    public void GetItem_OutOfRangeLower_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var dataset = CreateDataset(10);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => dataset.GetItem(-11));
    }

    [Fact]
    public void GetItem_OutOfRangeUpper_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var dataset = CreateDataset(10);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => dataset.GetItem(10));
    }

    [Fact]
    public void GetItem_MultipleCalls_SameItem()
    {
        // Arrange
        var dataset = CreateDataset(10);
        int index = 5;

        // Act
        var item1 = dataset.GetItem(index);
        var item2 = dataset.GetItem(index);

        // Assert
        Assert.Equal(item1, item2);
    }

    #endregion

    #region GetBatch Method Tests

    [Fact]
    public void GetBatch_ValidIndices_ReturnsCorrectItems()
    {
        // Arrange
        var dataset = CreateDataset(10);
        int[] indices = { 0, 2, 5, 9 };

        // Act
        var batch = dataset.GetBatch(indices);

        // Assert
        Assert.Equal(4, batch.Length);
        Assert.Equal(Convert.ChangeType(0, typeof(T)), batch[0]);
        Assert.Equal(Convert.ChangeType(2, typeof(T)), batch[1]);
        Assert.Equal(Convert.ChangeType(5, typeof(T)), batch[2]);
        Assert.Equal(Convert.ChangeType(9, typeof(T)), batch[3]);
    }

    [Fact]
    public void GetBatch_EmptyArray_ReturnsEmptyArray()
    {
        // Arrange
        var dataset = CreateDataset(10);
        int[] indices = Array.Empty<int>();

        // Act
        var batch = dataset.GetBatch(indices);

        // Assert
        Assert.Empty(batch);
    }

    [Fact]
    public void GetBatch_Null_ThrowsArgumentNullException()
    {
        // Arrange
        var dataset = CreateDataset(10);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => dataset.GetBatch(null));
    }

    [Fact]
    public void GetBatch_WithNegativeIndices_ReturnsCorrectItems()
    {
        // Arrange
        var dataset = CreateDataset(10);
        int[] indices = { -10, -5, -1 };

        // Act
        var batch = dataset.GetBatch(indices);

        // Assert
        Assert.Equal(3, batch.Length);
        Assert.Equal(default, batch[0]);
        Assert.Equal(Convert.ChangeType(5, typeof(T)), batch[1]);
        Assert.Equal(Convert.ChangeType(9, typeof(T)), batch[2]);
    }

    #endregion

    #region Thread Safety Tests

    [Fact]
    public async Task ConcurrentGetItem_MultipleThreads_NoErrors()
    {
        // Arrange
        var dataset = CreateDataset(100);
        int threadCount = 10;
        int readsPerThread = 1000;

        // Act
        var tasks = new Task[threadCount];
        for (int i = 0; i < threadCount; i++)
        {
            tasks[i] = Task.Run(() =>
            {
                for (int j = 0; j < readsPerThread; j++)
                {
                    var index = j % 100;
                    _ = dataset.GetItem(index);
                }
            });
        }
        await Task.WhenAll(tasks);

        // Assert - No exceptions thrown
    }

    [Fact]
    public async Task ConcurrentGetItem_MultipleThreads_CorrectResults()
    {
        // Arrange
        var dataset = CreateDataset(100);
        int threadCount = 10;
        int readsPerThread = 1000;

        // Act
        var tasks = new Task[threadCount];
        for (int i = 0; i < threadCount; i++)
        {
            tasks[i] = Task.Run(() =>
            {
                for (int j = 0; j < readsPerThread; j++)
                {
                    var index = j % 100;
                    var item = dataset.GetItem(index);
                    Assert.Equal(Convert.ChangeType(index, typeof(T)), item);
                }
            });
        }
        await Task.WhenAll(tasks);

        // Assert - All results verified in the tasks
    }

    [Fact]
    public async Task ConcurrentGetItem_HighContention_RaceFree()
    {
        // Arrange
        var dataset = CreateDataset(100);
        int threadCount = 100;
        int readsPerThread = 100;

        // Act
        var tasks = new Task[threadCount];
        for (int i = 0; i < threadCount; i++)
        {
            tasks[i] = Task.Run(() =>
            {
                for (int j = 0; j < readsPerThread; j++)
                {
                    var index = j % 100;
                    _ = dataset.GetItem(index);
                }
            });
        }
        await Task.WhenAll(tasks);

        // Assert - No exceptions thrown
    }

    #endregion
}

/// <summary>
/// Tests for the IDataset interface using integer datasets.
/// </summary>
public class DatasetTests : DatasetTestBase<IDataset<int>, int>
{
    protected override IDataset<int> CreateDataset(int count)
    {
        return new ArrayDataset<int>(CreateTestData(count));
    }

    protected override IDataset<int> CreateDataset(int[] items)
    {
        return new ArrayDataset<int>(items);
    }

    #region Edge Case Tests

    [Fact]
    public void EmptyDataset_CountZero()
    {
        // Arrange & Act
        var dataset = CreateDataset(0);

        // Assert
        Assert.Equal(0, dataset.Count);
    }

    [Fact]
    public void EmptyDataset_GetItem_Throws()
    {
        // Arrange
        var dataset = CreateDataset(0);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => dataset.GetItem(0));
    }

    [Fact]
    public void SingleItemDataset_CountOne()
    {
        // Arrange & Act
        var dataset = CreateDataset(1);

        // Assert
        Assert.Equal(1, dataset.Count);
    }

    [Fact]
    public void SingleItemDataset_GetItemZero_ReturnsItem()
    {
        // Arrange
        var dataset = CreateDataset(1);

        // Act
        var item = dataset.GetItem(0);

        // Assert
        Assert.Equal(0, item);
    }

    [Fact]
    public void DatasetWithNullItems_GetItem_ReturnsNull()
    {
        // Arrange
        var dataset = CreateDataset(new int?[] { 1, null, 3 });

        // Act
        var item = dataset.GetItem(1);

        // Assert
        Assert.Null(item);
    }

    [Fact]
    public void DatasetWithNullItems_Count_IncludesNulls()
    {
        // Arrange
        var dataset = CreateDataset(new int?[] { 1, null, 3, null });

        // Act
        var count = dataset.Count;

        // Assert
        Assert.Equal(4, count);
    }

    #endregion
}
