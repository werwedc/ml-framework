using MLFramework.Data;
using Xunit;

namespace MLFramework.Tests.Data;

/// <summary>
/// Tests for MapStyleDataset base class.
/// </summary>
public class MapStyleDatasetTests
{
    private class SimpleMapDataset : MapStyleDataset<int>
    {
        private readonly int[] _data;

        public SimpleMapDataset(int[] data)
        {
            _data = data;
        }

        public override int GetItem(int index) => _data[index];

        public override int Length => _data.Length;
    }

    [Fact]
    public void GetItem_ReturnsCorrectItem()
    {
        // Arrange
        var data = new[] { 10, 20, 30, 40, 50 };
        var dataset = new SimpleMapDataset(data);

        // Act & Assert
        Assert.Equal(10, dataset.GetItem(0));
        Assert.Equal(30, dataset.GetItem(2));
        Assert.Equal(50, dataset.GetItem(4));
    }

    [Fact]
    public void Length_ReturnsCorrectLength()
    {
        // Arrange
        var data = new[] { 1, 2, 3 };
        var dataset = new SimpleMapDataset(data);

        // Act & Assert
        Assert.Equal(3, dataset.Length);
    }

    [Fact]
    public void GetValidatedItem_ValidIndex_ReturnsItem()
    {
        // Arrange
        var data = new[] { 100, 200, 300 };
        var dataset = new SimpleMapDataset(data);

        // Act & Assert
        Assert.Equal(200, dataset.GetValidatedItem(1));
    }

    [Fact]
    public void GetValidatedItem_InvalidIndex_ThrowsException()
    {
        // Arrange
        var data = new[] { 1, 2, 3 };
        var dataset = new SimpleMapDataset(data);

        // Act & Assert
        Assert.Throws<IndexOutOfRangeException>(() => dataset.GetValidatedItem(-1));
        Assert.Throws<IndexOutOfRangeException>(() => dataset.GetValidatedItem(3));
        Assert.Throws<IndexOutOfRangeException>(() => dataset.GetValidatedItem(100));
    }

    [Fact]
    public void OnDatasetCreated_CalledDuringInitialization()
    {
        // Arrange
        bool onCreatedCalled = false;

        var dataset = new SimpleMapDataset(new[] { 1, 2, 3 })
        {
            OnDatasetCreated = () => { onCreatedCalled = true; }
        };

        // Act & Assert
        Assert.True(onCreatedCalled);
    }

    [Fact]
    public void EmptyDataset_ReturnsZeroLength()
    {
        // Arrange
        var dataset = new SimpleMapDataset(Array.Empty<int>());

        // Act & Assert
        Assert.Equal(0, dataset.Length);
    }

    [Fact]
    public void LargeDataset_HandlesCorrectly()
    {
        // Arrange
        var largeData = Enumerable.Range(0, 10000).ToArray();
        var dataset = new SimpleMapDataset(largeData);

        // Act & Assert
        Assert.Equal(10000, dataset.Length);
        Assert.Equal(0, dataset.GetItem(0));
        Assert.Equal(5000, dataset.GetItem(5000));
        Assert.Equal(9999, dataset.GetItem(9999));
    }
}
