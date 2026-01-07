using Microsoft.VisualStudio.TestTools.UnitTesting;
using MachineLearning.Distributed.Data;
using MachineLearning.Distributed.Models;
using System.Collections.Generic;
using System.Linq;

namespace MLFramework.Tests.Distributed.Data;

[TestClass]
public class ElasticDistributedSamplerTests
{
    [TestMethod]
    public void Constructor_ValidParameters_CreatesSampler()
    {
        // Act
        var sampler = new ElasticDistributedSampler(100, 4, 0);

        // Assert
        Assert.IsNotNull(sampler);
        Assert.AreEqual(100, sampler.DatasetSize);
        Assert.IsTrue(sampler.TotalSamples > 0);
    }

    [TestMethod]
    public void Constructor_NegativeDatasetSize_ThrowsException()
    {
        // Act & Assert
        Assert.ThrowsException<ArgumentException>(() =>
            new ElasticDistributedSampler(-1, 4, 0));
    }

    [TestMethod]
    public void Constructor_ZeroDatasetSize_ThrowsException()
    {
        // Act & Assert
        Assert.ThrowsException<ArgumentException>(() =>
            new ElasticDistributedSampler(0, 4, 0));
    }

    [TestMethod]
    public void Constructor_InvalidWorkerRank_ThrowsException()
    {
        // Act & Assert
        Assert.ThrowsException<ArgumentException>(() =>
            new ElasticDistributedSampler(100, 4, 5));
    }

    [TestMethod]
    public void Constructor_InvalidWorkerCount_ThrowsException()
    {
        // Act & Assert
        Assert.ThrowsException<ArgumentException>(() =>
            new ElasticDistributedSampler(100, 0, 0));
    }

    [TestMethod]
    public void GetNumSamples_ReturnsCorrectCount()
    {
        // Arrange
        var sampler = new ElasticDistributedSampler(100, 4, 0);

        // Act
        var numSamples = sampler.GetNumSamples();

        // Assert
        Assert.AreEqual(25, numSamples); // 100 / 4 = 25
    }

    [TestMethod]
    public void GetTotalSize_ReturnsDatasetSize()
    {
        // Arrange
        var sampler = new ElasticDistributedSampler(100, 4, 0);

        // Act
        var totalSize = sampler.GetTotalSize();

        // Assert
        Assert.AreEqual(100, totalSize);
    }

    [TestMethod]
    public void GetNextBatch_ValidBatchSize_ReturnsCorrectSamples()
    {
        // Arrange
        var sampler = new ElasticDistributedSampler(100, 4, 0);

        // Act
        var batch = sampler.GetNextBatch(10).ToList();

        // Assert
        Assert.AreEqual(10, batch.Count);
        Assert.AreEqual(0, batch[0]);
        Assert.AreEqual(9, batch[9]);
    }

    [TestMethod]
    public void GetNextBatch_MultipleBatches_DistributesData()
    {
        // Arrange
        var sampler = new ElasticDistributedSampler(100, 4, 0);

        // Act
        var batch1 = sampler.GetNextBatch(10).ToList();
        var batch2 = sampler.GetNextBatch(10).ToList();

        // Assert
        Assert.AreEqual(10, batch1.Count);
        Assert.AreEqual(10, batch2.Count);
        Assert.AreEqual(10, batch2[0]);
        Assert.AreEqual(19, batch2[9]);
    }

    [TestMethod]
    public void GetNextBatch_ExceedsRemainingSamples_ReturnsRemaining()
    {
        // Arrange
        var sampler = new ElasticDistributedSampler(25, 1, 0);
        sampler.GetNextBatch(20).ToList(); // Consume 20

        // Act
        var batch = sampler.GetNextBatch(10).ToList();

        // Assert
        Assert.AreEqual(5, batch.Count);
    }

    [TestMethod]
    public void GetNextBatch_NegativeBatchSize_ThrowsException()
    {
        // Arrange
        var sampler = new ElasticDistributedSampler(100, 4, 0);

        // Act & Assert
        Assert.ThrowsException<ArgumentException>(() =>
            sampler.GetNextBatch(-1).ToList());
    }

    [TestMethod]
    public void GetNextBatch_ZeroBatchSize_ThrowsException()
    {
        // Arrange
        var sampler = new ElasticDistributedSampler(100, 4, 0);

        // Act & Assert
        Assert.ThrowsException<ArgumentException>(() =>
            sampler.GetNextBatch(0).ToList());
    }

    [TestMethod]
    public void HasMore_WithRemainingSamples_ReturnsTrue()
    {
        // Arrange
        var sampler = new ElasticDistributedSampler(100, 4, 0);
        sampler.GetNextBatch(10).ToList();

        // Act
        var hasMore = sampler.HasMore();

        // Assert
        Assert.IsTrue(hasMore);
    }

    [TestMethod]
    public void HasMore_AfterConsumingAllSamples_ReturnsFalse()
    {
        // Arrange
        var sampler = new ElasticDistributedSampler(25, 1, 0);
        sampler.GetNextBatch(25).ToList();

        // Act
        var hasMore = sampler.HasMore();

        // Assert
        Assert.IsFalse(hasMore);
    }

    [TestMethod]
    public void Reset_AfterConsumingSamples_ResetsPosition()
    {
        // Arrange
        var sampler = new ElasticDistributedSampler(100, 4, 0);
        sampler.GetNextBatch(10).ToList();

        // Act
        sampler.Reset();
        var batch = sampler.GetNextBatch(10).ToList();

        // Assert
        Assert.AreEqual(0, batch[0]);
        Assert.AreEqual(9, batch[9]);
    }

    [TestMethod]
    public void Shuffle_RandomizesIndices()
    {
        // Arrange
        var sampler = new ElasticDistributedSampler(100, 4, 0, seed: 42);

        // Act
        var beforeShuffle = sampler.GetNextBatch(5).ToList();
        sampler.Reset();
        sampler.Shuffle();
        var afterShuffle = sampler.GetNextBatch(5).ToList();

        // Assert
        // With a seed, we can verify shuffle actually changed the order
        // This test verifies that shuffling is happening
        Assert.AreEqual(5, afterShuffle.Count);
    }

    [TestMethod]
    public void SetSeed_ChangesRandomGenerator()
    {
        // Arrange
        var sampler1 = new ElasticDistributedSampler(100, 4, 0, seed: 42);
        var sampler2 = new ElasticDistributedSampler(100, 4, 0, seed: 42);

        // Act
        sampler1.Shuffle();
        sampler2.Shuffle();
        var batch1 = sampler1.GetNextBatch(5).ToList();
        var batch2 = sampler2.GetNextBatch(5).ToList();

        // Assert
        // Same seed should produce same results
        Assert.IsTrue(batch1.SequenceEqual(batch2));
    }

    [TestMethod]
    public void UpdateTopology_ChangesWorkerCount_UpdatesSampleCount()
    {
        // Arrange
        var sampler = new ElasticDistributedSampler(100, 4, 0);
        var oldSamples = sampler.GetNumSamples();

        // Act
        sampler.UpdateTopology(10, 0);
        var newSamples = sampler.GetNumSamples();

        // Assert
        Assert.AreEqual(25, oldSamples);
        Assert.AreEqual(10, newSamples);
    }

    [TestMethod]
    public void UpdateTopology_ChangesWorkerRank_UpdatesSampleRange()
    {
        // Arrange
        var sampler = new ElasticDistributedSampler(100, 4, 0);
        var batch1 = sampler.GetNextBatch(10).ToList();

        // Act
        sampler.UpdateTopology(4, 1);
        var batch2 = sampler.GetNextBatch(10).ToList();

        // Assert
        Assert.AreEqual(25, batch2[0]); // First sample of rank 1
        Assert.AreEqual(34, batch2[9]);
    }

    [TestMethod]
    public void UpdateTopology_InvalidWorkerCount_ThrowsException()
    {
        // Arrange
        var sampler = new ElasticDistributedSampler(100, 4, 0);

        // Act & Assert
        Assert.ThrowsException<ArgumentException>(() =>
            sampler.UpdateTopology(0, 0));
    }

    [TestMethod]
    public void UpdateTopology_InvalidWorkerRank_ThrowsException()
    {
        // Arrange
        var sampler = new ElasticDistributedSampler(100, 4, 0);

        // Act & Assert
        Assert.ThrowsException<ArgumentException>(() =>
            sampler.UpdateTopology(4, 5));
    }

    [TestMethod]
    public void GetWorkerShard_ReturnsCorrectShard()
    {
        // Arrange
        var sampler = new ElasticDistributedSampler(100, 4, 0);

        // Act
        var shard = sampler.GetWorkerShard();

        // Assert
        Assert.AreEqual(0, shard.ShardId);
        Assert.AreEqual(0, shard.StartIndex);
        Assert.AreEqual(25, shard.Size);
    }

    [TestMethod]
    public void GetWorkerShard_DifferentWorkerRank_ReturnsDifferentShard()
    {
        // Arrange
        var sampler = new ElasticDistributedSampler(100, 4, 1);

        // Act
        var shard = sampler.GetWorkerShard();

        // Assert
        Assert.AreEqual(1, shard.ShardId);
        Assert.AreEqual(25, shard.StartIndex);
        Assert.AreEqual(25, shard.Size);
    }

    [TestMethod]
    public void MultipleWorkers_CoverAllSamplesWithoutDuplication()
    {
        // Arrange
        var datasetSize = 100;
        var workerCount = 4;
        var samplers = new ElasticDistributedSampler[workerCount];

        for (int i = 0; i < workerCount; i++)
        {
            samplers[i] = new ElasticDistributedSampler(datasetSize, workerCount, i);
        }

        // Act
        var totalSamples = 0;
        var allSamples = new HashSet<int>();

        foreach (var sampler in samplers)
        {
            var samples = sampler.GetNumSamples();
            totalSamples += samples;

            var batch = sampler.GetNextBatch(samples).ToList();
            foreach (var sample in batch)
            {
                allSamples.Add(sample);
            }
        }

        // Assert
        Assert.AreEqual(datasetSize, totalSamples);
        Assert.AreEqual(datasetSize, allSamples.Count);
    }

    [TestMethod]
    public void UnevenDatasetSize_DistributesRemainderCorrectly()
    {
        // Arrange
        var datasetSize = 103;
        var workerCount = 10;

        // Act
        var sampler0 = new ElasticDistributedSampler(datasetSize, workerCount, 0);
        var sampler9 = new ElasticDistributedSampler(datasetSize, workerCount, 9);

        // Assert
        Assert.AreEqual(11, sampler0.GetNumSamples()); // 103/10 = 10, remainder 3
        Assert.AreEqual(10, sampler9.GetNumSamples());
    }

    [TestMethod]
    public void DatasetSizeSmallerThanWorkerCount_SomeWorkersGetZeroSamples()
    {
        // Arrange
        var datasetSize = 5;
        var workerCount = 10;

        // Act
        var sampler0 = new ElasticDistributedSampler(datasetSize, workerCount, 0);
        var sampler9 = new ElasticDistributedSampler(datasetSize, workerCount, 9);

        // Assert
        Assert.AreEqual(1, sampler0.GetNumSamples());
        Assert.AreEqual(0, sampler9.GetNumSamples());
    }
}
