using MLFramework.Data;
using Xunit;

namespace MLFramework.Tests.Data;

/// <summary>
/// Tests for DataLoader class.
/// </summary>
public class DataLoaderTests
{
    private class SimpleDataset : MapStyleDataset<int>
    {
        private readonly int[] _data;

        public SimpleDataset(int[] data)
        {
            _data = data;
        }

        public override int GetItem(int index) => _data[index];
        public override int Length => _data.Length;
    }

    [Fact]
    public void Constructor_NullDataset_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new DataLoader<int>(null, 32));
    }

    [Fact]
    public void Constructor_NonPositiveBatchSize_ThrowsException()
    {
        // Arrange
        var dataset = new SimpleDataset(new[] { 1, 2, 3 });

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => new DataLoader<int>(dataset, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DataLoader<int>(dataset, -1));
    }

    [Fact]
    public void Constructor_EmptyDatasetWithDropLast_ThrowsException()
    {
        // Arrange
        var dataset = new SimpleDataset(Array.Empty<int>());

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => new DataLoader<int>(dataset, 32, dropLast: true));
    }

    [Fact]
    public void Constructor_EmptyDatasetNoDropLast_DoesNotThrow()
    {
        // Arrange
        var dataset = new SimpleDataset(Array.Empty<int>());

        // Act & Assert
        var exception = Record.Exception(() => new DataLoader<int>(dataset, 32, dropLast: false));
        Assert.Null(exception);
    }

    [Fact]
    public void Dataset_ReturnsOriginalDataset()
    {
        // Arrange
        var dataset = new SimpleDataset(new[] { 1, 2, 3 });
        var dataloader = new DataLoader<int>(dataset, 2);

        // Act & Assert
        Assert.Same(dataset, dataloader.Dataset);
    }

    [Fact]
    public void BatchSize_ReturnsCorrectValue()
    {
        // Arrange
        var dataset = new SimpleDataset(new[] { 1, 2, 3, 4, 5 });
        var dataloader = new DataLoader<int>(dataset, 3);

        // Act & Assert
        Assert.Equal(3, dataloader.BatchSize);
    }

    [Fact]
    public void DatasetLength_ReturnsDatasetLength()
    {
        // Arrange
        var dataset = new SimpleDataset(new[] { 1, 2, 3, 4, 5 });
        var dataloader = new DataLoader<int>(dataset, 2);

        // Act & Assert
        Assert.Equal(5, dataloader.DatasetLength);
    }

    [Fact]
    public void NumBatches_ExactMultiple_ReturnsCorrectCount()
    {
        // Arrange
        var dataset = new SimpleDataset(Enumerable.Range(0, 100).ToArray());
        var dataloader = new DataLoader<int>(dataset, 25, dropLast: false);

        // Act & Assert
        Assert.Equal(4, dataloader.NumBatches);
    }

    [Fact]
    public void NumBatches_WithRemainderNoDropLast_ReturnsCorrectCount()
    {
        // Arrange
        var dataset = new SimpleDataset(Enumerable.Range(0, 100).ToArray());
        var dataloader = new DataLoader<int>(dataset, 32, dropLast: false);

        // Act & Assert
        Assert.Equal(4, dataloader.NumBatches); // 3 full + 1 partial = 4
    }

    [Fact]
    public void NumBatches_WithRemainderDropLast_ReturnsCorrectCount()
    {
        // Arrange
        var dataset = new SimpleDataset(Enumerable.Range(0, 100).ToArray());
        var dataloader = new DataLoader<int>(dataset, 32, dropLast: true);

        // Act & Assert
        Assert.Equal(3, dataloader.NumBatches); // 3 full batches, last dropped
    }

    [Fact]
    public void GetEnumerator_ReturnsCorrectBatches()
    {
        // Arrange
        var dataset = new SimpleDataset(Enumerable.Range(0, 10).ToArray());
        var dataloader = new DataLoader<int>(dataset, batchSize: 3, dropLast: false);

        // Act
        var batches = dataloader.ToList();

        // Assert
        Assert.Equal(4, batches.Count);
        Assert.Equal(new[] { 0, 1, 2 }, (int[])batches[0]);
        Assert.Equal(new[] { 3, 4, 5 }, (int[])batches[1]);
        Assert.Equal(new[] { 6, 7, 8 }, (int[])batches[2]);
        Assert.Equal(new[] { 9 }, (int[])batches[3]);
    }

    [Fact]
    public void GetEnumerator_DropLast_DropsIncompleteBatch()
    {
        // Arrange
        var dataset = new SimpleDataset(Enumerable.Range(0, 10).ToArray());
        var dataloader = new DataLoader<int>(dataset, batchSize: 3, dropLast: true);

        // Act
        var batches = dataloader.ToList();

        // Assert
        Assert.Equal(3, batches.Count);
        Assert.All(batches, batch => Assert.Equal(3, ((int[])batch).Length));
    }

    [Fact]
    public void GetEnumerator_ShuflleFalse_SequentialOrder()
    {
        // Arrange
        var dataset = new SimpleDataset(Enumerable.Range(0, 10).ToArray());
        var dataloader = new DataLoader<int>(dataset, batchSize: 5, shuffle: false);

        // Act
        var batches = dataloader.ToList();

        // Assert
        Assert.Equal(new[] { 0, 1, 2, 3, 4 }, (int[])batches[0]);
        Assert.Equal(new[] { 5, 6, 7, 8, 9 }, (int[])batches[1]);
    }

    [Fact]
    public void GetEnumerator_ShuffleTrue_RandomOrder()
    {
        // Arrange
        var dataset = new SimpleDataset(Enumerable.Range(0, 20).ToArray());
        var dataloader = new DataLoader<int>(dataset, batchSize: 5, shuffle: true, dropLast: false);

        // Act
        var batches = dataloader.ToList();
        var allIndices = batches.SelectMany(b => (int[])b).ToList();

        // Assert
        Assert.Equal(20, allIndices.Count);
        Assert.Equal(20, allIndices.Distinct().Count()); // All indices present
        Assert.NotEqual(Enumerable.Range(0, 20).ToList(), allIndices); // Not sequential
    }

    [Fact]
    public void GetEnumerator_CustomSampler_UsesProvidedSampler()
    {
        // Arrange
        var dataset = new SimpleDataset(Enumerable.Range(0, 10).ToArray());
        var customSampler = new SequentialSampler(10);
        var dataloader = new DataLoader<int>(dataset, batchSize: 3, sampler: customSampler);

        // Act
        var batches = dataloader.ToList();

        // Assert
        Assert.Equal(4, batches.Count);
        Assert.Equal(new[] { 0, 1, 2 }, (int[])batches[0]);
    }

    [Fact]
    public void GetEnumerator_CustomBatchSampler_UsesProvidedBatchSampler()
    {
        // Arrange
        var dataset = new SimpleDataset(Enumerable.Range(0, 10).ToArray());
        var sampler = new SequentialSampler(10);
        var batchSampler = new BatchSampler(sampler, batchSize: 3, dropLast: false);
        var dataloader = new DataLoader<int>(dataset, batchSize: 5, batchSampler: batchSampler);

        // Act
        var batches = dataloader.ToList();

        // Assert
        Assert.Equal(4, batches.Count); // batchSampler determines batch count, not batchSize parameter
        Assert.Equal(3, ((int[])batches[0]).Length);
    }

    [Fact]
    public void GetEnumerator_CustomCollateFunction_UsesProvidedFunction()
    {
        // Arrange
        var dataset = new SimpleDataset(Enumerable.Range(0, 10).ToArray());
        var dataloader = new DataLoader<int>(dataset, batchSize: 3,
            collateFn: batch => batch.Sum()); // Sum the batch

        // Act
        var batches = dataloader.ToList();

        // Assert
        Assert.Equal(3, batches[0]); // 0 + 1 + 2 = 3
        Assert.Equal(12, batches[1]); // 3 + 4 + 5 = 12
        Assert.Equal(21, batches[2]); // 6 + 7 + 8 = 21
        Assert.Equal(9, batches[3]); // 9 = 9
    }

    [Fact]
    public void GetEnumerator_DefaultCollate_ReturnsBatchArray()
    {
        // Arrange
        var dataset = new SimpleDataset(Enumerable.Range(0, 5).ToArray());
        var dataloader = new DataLoader<int>(dataset, batchSize: 2);

        // Act
        var batches = dataloader.ToList();

        // Assert
        Assert.IsType<int[]>(batches[0]);
        Assert.Equal(new[] { 0, 1 }, (int[])batches[0]);
    }

    [Fact]
    public void GetEnumerator_EmptyDataset_ReturnsEmpty()
    {
        // Arrange
        var dataset = new SimpleDataset(Array.Empty<int>());
        var dataloader = new DataLoader<int>(dataset, batchSize: 32, dropLast: false);

        // Act
        var batches = dataloader.ToList();

        // Assert
        Assert.Empty(batches);
    }

    [Fact]
    public void GetEnumerator_SingleItemDataset_ReturnsSingleBatch()
    {
        // Arrange
        var dataset = new SimpleDataset(new[] { 42 });
        var dataloader = new DataLoader<int>(dataset, batchSize: 32, dropLast: false);

        // Act
        var batches = dataloader.ToList();

        // Assert
        Assert.Single(batches);
        Assert.Single((int[])batches[0]);
        Assert.Equal(42, ((int[])batches[0])[0]);
    }

    [Fact]
    public void GetEnumerator_BatchSizeOne_ReturnsAllItems()
    {
        // Arrange
        var dataset = new SimpleDataset(Enumerable.Range(0, 10).ToArray());
        var dataloader = new DataLoader<int>(dataset, batchSize: 1);

        // Act
        var batches = dataloader.ToList();

        // Assert
        Assert.Equal(10, batches.Count);
        for (int i = 0; i < 10; i++)
        {
            Assert.Single((int[])batches[i]);
            Assert.Equal(i, ((int[])batches[i])[0]);
        }
    }

    [Fact]
    public void GetEnumerator_LargeDataset_HandlesCorrectly()
    {
        // Arrange
        var dataset = new SimpleDataset(Enumerable.Range(0, 10000).ToArray());
        var dataloader = new DataLoader<int>(dataset, batchSize: 100, dropLast: false);

        // Act
        var batches = dataloader.ToList();

        // Assert
        Assert.Equal(100, batches.Count);
        Assert.All(batches, batch => Assert.Equal(100, ((int[])batch).Length));
    }

    [Fact]
    public void GetEnumerator_MultipleIterations_Independent()
    {
        // Arrange
        var dataset = new SimpleDataset(Enumerable.Range(0, 10).ToArray());
        var dataloader = new DataLoader<int>(dataset, batchSize: 3, shuffle: true);

        // Act
        var first = dataloader.ToList();
        var second = dataloader.ToList();

        // Assert - With shuffle, should be different order (very unlikely to be same by chance)
        Assert.NotEqual(first, second);
    }

    [Fact]
    public void GetEnumerator_WithForeach_WorksCorrectly()
    {
        // Arrange
        var dataset = new SimpleDataset(Enumerable.Range(0, 10).ToArray());
        var dataloader = new DataLoader<int>(dataset, batchSize: 3);
        var count = 0;

        // Act
        foreach (var batch in dataloader)
        {
            count++;
        }

        // Assert
        Assert.Equal(4, count);
    }

    [Fact]
    public void GetEnumerator_WithLINQ_WorksCorrectly()
    {
        // Arrange
        var dataset = new SimpleDataset(Enumerable.Range(1, 10).ToArray());
        var dataloader = new DataLoader<int>(dataset, batchSize: 3);

        // Act
        var batchCount = dataloader.Count();

        // Assert
        Assert.Equal(4, batchCount);
    }
}
