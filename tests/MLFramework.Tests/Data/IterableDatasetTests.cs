using MLFramework.Data;
using Xunit;

namespace MLFramework.Tests.Data;

/// <summary>
/// Tests for IterableDataset base class.
/// </summary>
public class IterableDatasetTests
{
    private class SimpleIterableDataset : IterableDataset<int>
    {
        private readonly IEnumerable<int> _data;

        public SimpleIterableDataset(IEnumerable<int> data)
        {
            _data = data;
        }

        public override IEnumerator<int> GetEnumerator()
        {
            return _data.GetEnumerator();
        }
    }

    [Fact]
    public void GetEnumerator_ReturnsAllItems()
    {
        // Arrange
        var data = new[] { 10, 20, 30, 40, 50 };
        var dataset = new SimpleIterableDataset(data);

        // Act
        var items = dataset.ToList();

        // Assert
        Assert.Equal(5, items.Count);
        Assert.Equal(10, items[0]);
        Assert.Equal(30, items[2]);
        Assert.Equal(50, items[4]);
    }

    [Fact]
    public void GetEnumerator_EmptyDataset_ReturnsEmpty()
    {
        // Arrange
        var dataset = new SimpleIterableDataset(Array.Empty<int>());

        // Act
        var items = dataset.ToList();

        // Assert
        Assert.Empty(items);
    }

    [Fact]
    public void GetEnumerator_MultipleIterations_Independent()
    {
        // Arrange
        var data = new[] { 1, 2, 3 };
        var dataset = new SimpleIterableDataset(data);

        // Act
        var firstIteration = dataset.ToList();
        var secondIteration = dataset.ToList();

        // Assert
        Assert.Equal(firstIteration, secondIteration);
    }

    [Fact]
    public void OnDatasetCreated_CalledDuringInitialization()
    {
        // Arrange
        bool onCreatedCalled = false;

        var dataset = new SimpleIterableDataset(new[] { 1, 2, 3 })
        {
            OnDatasetCreated = () => { onCreatedCalled = true; }
        };

        // Act & Assert
        Assert.True(onCreatedCalled);
    }

    [Fact]
    public void GetEnumerator_LargeDataset_HandlesCorrectly()
    {
        // Arrange
        var largeData = Enumerable.Range(0, 10000);
        var dataset = new SimpleIterableDataset(largeData);

        // Act
        var items = dataset.ToList();

        // Assert
        Assert.Equal(10000, items.Count);
        Assert.Equal(0, items[0]);
        Assert.Equal(5000, items[5000]);
        Assert.Equal(9999, items[9999]);
    }

    [Fact]
    public void GetEnumerator_StreamingData_SupportsLazyEvaluation()
    {
        // Arrange
        int generationCount = 0;
        IEnumerable<int> StreamingData()
        {
            for (int i = 0; i < 10; i++)
            {
                generationCount++;
                yield return i;
            }
        }

        var dataset = new SimpleIterableDataset(StreamingData());

        // Act
        var enumerator = dataset.GetEnumerator();
        enumerator.MoveNext(); // Only consume first item

        // Assert
        Assert.Equal(1, generationCount); // Only first item was generated
    }
}
