using MLFramework.Data;
using Xunit;

namespace MLFramework.Tests.Data;

/// <summary>
/// Tests for BatchSampler.
/// </summary>
public class BatchSamplerTests
{
    [Fact]
    public void Iterate_ExactMultiple_ReturnsFullBatches()
    {
        // Arrange
        var sampler = new SequentialSampler(10);
        var batchSampler = new BatchSampler(sampler, batchSize: 3, dropLast: false);

        // Act
        var batches = batchSampler.Iterate().ToList();

        // Assert
        Assert.Equal(4, batches.Count); // 3, 3, 3, 1
        Assert.Equal(3, batches[0].Length);
        Assert.Equal(3, batches[1].Length);
        Assert.Equal(3, batches[2].Length);
        Assert.Equal(1, batches[3].Length);
    }

    [Fact]
    public void Iterate_DropLast_ExactMultiple_ReturnsFullBatches()
    {
        // Arrange
        var sampler = new SequentialSampler(9);
        var batchSampler = new BatchSampler(sampler, batchSize: 3, dropLast: true);

        // Act
        var batches = batchSampler.Iterate().ToList();

        // Assert
        Assert.Equal(3, batches.Count);
        Assert.All(batches, batch => Assert.Equal(3, batch.Length));
    }

    [Fact]
    public void Iterate_DropLast_Remainder_DropsLast()
    {
        // Arrange
        var sampler = new SequentialSampler(10);
        var batchSampler = new BatchSampler(sampler, batchSize: 3, dropLast: true);

        // Act
        var batches = batchSampler.Iterate().ToList();

        // Assert
        Assert.Equal(3, batches.Count);
        Assert.All(batches, batch => Assert.Equal(3, batch.Length));
    }

    [Fact]
    public void Iterate_DropLastFalse_IncludesLastPartial()
    {
        // Arrange
        var sampler = new SequentialSampler(10);
        var batchSampler = new BatchSampler(sampler, batchSize: 3, dropLast: false);

        // Act
        var batches = batchSampler.Iterate().ToList();

        // Assert
        Assert.Equal(4, batches.Count);
        Assert.Equal(1, batches.Last().Length);
    }

    [Fact]
    public void BatchSize_ReturnsCorrectValue()
    {
        // Arrange
        var sampler = new SequentialSampler(10);
        var batchSampler = new BatchSampler(sampler, batchSize: 5, dropLast: false);

        // Act & Assert
        Assert.Equal(5, batchSampler.BatchSize);
    }

    [Fact]
    public void Constructor_NullSampler_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new BatchSampler(null, batchSize: 3));
    }

    [Fact]
    public void Constructor_NonPositiveBatchSize_ThrowsException()
    {
        var sampler = new SequentialSampler(10);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => new BatchSampler(sampler, batchSize: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new BatchSampler(sampler, batchSize: -1));
    }

    [Fact]
    public void Iterate_EmptyDataset_ReturnsEmpty()
    {
        // Arrange
        var sampler = new SequentialSampler(0);
        var batchSampler = new BatchSampler(sampler, batchSize: 3, dropLast: false);

        // Act
        var batches = batchSampler.Iterate().ToList();

        // Assert
        Assert.Empty(batches);
    }

    [Fact]
    public void Iterate_SingleItem_NoDropLast_ReturnsSingleBatch()
    {
        // Arrange
        var sampler = new SequentialSampler(1);
        var batchSampler = new BatchSampler(sampler, batchSize: 3, dropLast: false);

        // Act
        var batches = batchSampler.Iterate().ToList();

        // Assert
        Assert.Single(batches);
        Assert.Single(batches[0]);
        Assert.Equal(0, batches[0][0]);
    }

    [Fact]
    public void Iterate_SingleItem_DropLast_ReturnsEmpty()
    {
        // Arrange
        var sampler = new SequentialSampler(1);
        var batchSampler = new BatchSampler(sampler, batchSize: 3, dropLast: true);

        // Act
        var batches = batchSampler.Iterate().ToList();

        // Assert
        Assert.Empty(batches);
    }

    [Fact]
    public void Iterate_BatchSizeLargerThanDataset_NoDropLast_ReturnsSingleBatch()
    {
        // Arrange
        var sampler = new SequentialSampler(5);
        var batchSampler = new BatchSampler(sampler, batchSize: 10, dropLast: false);

        // Act
        var batches = batchSampler.Iterate().ToList();

        // Assert
        Assert.Single(batches);
        Assert.Equal(5, batches[0].Length);
    }

    [Fact]
    public void Iterate_BatchSizeLargerThanDataset_DropLast_ReturnsEmpty()
    {
        // Arrange
        var sampler = new SequentialSampler(5);
        var batchSampler = new BatchSampler(sampler, batchSize: 10, dropLast: true);

        // Act
        var batches = batchSampler.Iterate().ToList();

        // Assert
        Assert.Empty(batches);
    }

    [Fact]
    public void Iterate_WithRandomSampler_RespectsShuffling()
    {
        // Arrange
        var sampler = new RandomSampler(9, seed: 42);
        var batchSampler = new BatchSampler(sampler, batchSize: 3, dropLast: false);

        // Act
        var batches = batchSampler.Iterate().ToList();

        // Assert
        Assert.Equal(3, batches.Count);
        Assert.All(batches, batch => Assert.Equal(3, batch.Length));

        // Verify that batches are shuffled (not sequential)
        var allIndices = batches.SelectMany(b => b).ToList();
        var firstBatchIsSequential = allIndices[0] == 0 && allIndices[1] == 1 && allIndices[2] == 2;
        Assert.False(firstBatchIsSequential);
    }

    [Fact]
    public void Iterate_WithSequentialSampler_PredictableBatches()
    {
        // Arrange
        var sampler = new SequentialSampler(10);
        var batchSampler = new BatchSampler(sampler, batchSize: 3, dropLast: false);

        // Act
        var batches = batchSampler.Iterate().ToList();

        // Assert
        Assert.Equal(new[] { 0, 1, 2 }, batches[0]);
        Assert.Equal(new[] { 3, 4, 5 }, batches[1]);
        Assert.Equal(new[] { 6, 7, 8 }, batches[2]);
        Assert.Equal(new[] { 9 }, batches[3]);
    }

    [Fact]
    public void Iterate_LargeDataset_HandlesCorrectly()
    {
        // Arrange
        var sampler = new SequentialSampler(10000);
        var batchSampler = new BatchSampler(sampler, batchSize: 100, dropLast: false);

        // Act
        var batches = batchSampler.Iterate().ToList();

        // Assert
        Assert.Equal(100, batches.Count);
        Assert.All(batches.Take(99), batch => Assert.Equal(100, batch.Length));
        Assert.Equal(100, batches.Last().Length); // 10000 / 100 = 100 exactly
    }

    [Fact]
    public void Iterate_MultipleIterations_Independent()
    {
        // Arrange
        var sampler = new SequentialSampler(5);
        var batchSampler = new BatchSampler(sampler, batchSize: 2, dropLast: false);

        // Act
        var first = batchSampler.Iterate().ToList();
        var second = batchSampler.Iterate().ToList();

        // Assert
        Assert.Equal(first, second);
    }
}
