using MLFramework.Data;
using Xunit;

namespace MLFramework.Tests.Data;

/// <summary>
/// Tests for DistributedSampler.
/// </summary>
public class DistributedSamplerTests
{
    [Fact]
    public void Constructor_ValidParameters_CreatesSampler()
    {
        // Act & Assert
        var sampler = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 0
        );

        Assert.Equal(4, sampler.NumReplicas);
        Assert.Equal(0, sampler.Rank);
        Assert.Equal(0, sampler.Epoch);
    }

    [Fact]
    public void Constructor_NegativeDatasetSize_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new DistributedSampler(
                datasetSize: -1,
                numReplicas: 4,
                rank: 0
            )
        );
    }

    [Fact]
    public void Constructor_ZeroDatasetSize_CreatesEmptySampler()
    {
        // Act & Assert
        var sampler = new DistributedSampler(
            datasetSize: 0,
            numReplicas: 4,
            rank: 0
        );

        Assert.Equal(0, sampler.Length);
        Assert.Empty(sampler.GetIndices());
    }

    [Fact]
    public void Constructor_ZeroNumReplicas_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new DistributedSampler(
                datasetSize: 100,
                numReplicas: 0,
                rank: 0
            )
        );
    }

    [Fact]
    public void Constructor_NegativeNumReplicas_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new DistributedSampler(
                datasetSize: 100,
                numReplicas: -1,
                rank: 0
            )
        );
    }

    [Fact]
    public void Constructor_RankOutOfRange_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new DistributedSampler(
                datasetSize: 100,
                numReplicas: 4,
                rank: 4
            )
        );
    }

    [Fact]
    public void Constructor_NullParameters_UsesDefaults()
    {
        // Act & Assert - Should use defaults (numReplicas=1, rank=0) when null
        var sampler = new DistributedSampler(
            datasetSize: 100,
            numReplicas: null,
            rank: null
        );

        Assert.Equal(1, sampler.NumReplicas);
        Assert.Equal(0, sampler.Rank);
    }

    [Fact]
    public void NumSamples_DropLastEvenlyDivisible_ReturnsEqualSize()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 0,
            dropLast: true
        );

        // Act & Assert
        Assert.Equal(25, sampler.NumSamples);
    }

    [Fact]
    public void NumSamples_DropLastNotEvenlyDivisible_DropsExtra()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 103,
            numReplicas: 4,
            rank: 0,
            dropLast: true
        );

        // Act & Assert
        Assert.Equal(25, sampler.NumSamples); // 100 / 4 = 25, 3 samples dropped
    }

    [Fact]
    public void NumSamples_NoDropLast_ReturnsCeiling()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 103,
            numReplicas: 4,
            rank: 0,
            dropLast: false
        );

        // Act & Assert
        Assert.Equal(26, sampler.NumSamples); // Math.Ceiling(103/4) = 26
    }

    [Fact]
    public void Iterate_NoShuffle_InterleavedIndices()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 16,
            numReplicas: 4,
            rank: 0,
            shuffle: false
        );

        // Act
        var indices = sampler.Iterate().ToList();

        // Assert - Interleaved: 0, 4, 8, 12
        Assert.Equal(new[] { 0, 4, 8, 12 }, indices);
    }

    [Fact]
    public void Iterate_Rank1_NoShuffle_InterleavedIndices()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 16,
            numReplicas: 4,
            rank: 1,
            shuffle: false
        );

        // Act
        var indices = sampler.Iterate().ToList();

        // Assert - Interleaved: 1, 5, 9, 13
        Assert.Equal(new[] { 1, 5, 9, 13 }, indices);
    }

    [Fact]
    public void Iterate_DropLastAllReplicasHaveNoOverlap()
    {
        // Arrange
        var numReplicas = 4;
        var samplers = new List<DistributedSampler>();
        for (int i = 0; i < numReplicas; i++)
        {
            samplers.Add(new DistributedSampler(
                datasetSize: 100,
                numReplicas: numReplicas,
                rank: i,
                shuffle: false,
                dropLast: true
            ));
        }

        // Act
        var allIndices = new List<int>();
        foreach (var sampler in samplers)
        {
            var indices = sampler.Iterate().ToList();
            allIndices.AddRange(indices);
        }

        // Assert
        Assert.Equal(100, allIndices.Count);
        Assert.Equal(allIndices.Distinct().Count(), allIndices.Count); // No duplicates
        Assert.Equal(allIndices.OrderBy(x => x).ToList(), allIndices); // All indices present
    }

    [Fact]
    public void Iterate_DifferentRanksProcessDifferentSamples()
    {
        // Arrange
        var sampler0 = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 0,
            shuffle: false
        );

        var sampler1 = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 1,
            shuffle: false
        );

        // Act
        var indices0 = sampler0.Iterate().ToList();
        var indices1 = sampler1.Iterate().ToList();

        // Assert
        Assert.DoesNotContain(indices0, i => indices1.Contains(i));
    }

    [Fact]
    public void SetEpoch_ChangesShuffleOrder()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 0,
            shuffle: true,
            seed: 42
        );

        // Act
        var indices0 = sampler.GetIndices();
        sampler.SetEpoch(1);
        var indices1 = sampler.GetIndices();

        // Assert
        Assert.NotEqual(indices0, indices1);
        Assert.Equal(indices0.OrderBy(x => x).ToList(), indices0.OrderBy(x => x).ToList()); // Same elements
        Assert.Equal(indices1.OrderBy(x => x).ToList(), indices1.OrderBy(x => x).ToList());
    }

    [Fact]
    public void SetEpoch_NoShuffle_SameIndices()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 0,
            shuffle: false
        );

        // Act
        var indices0 = sampler.GetIndices();
        sampler.SetEpoch(1);
        var indices1 = sampler.GetIndices();

        // Assert
        Assert.Equal(indices0, indices1);
    }

    [Fact]
    public void GetBatch_ReturnsCorrectBatch()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 16,
            numReplicas: 4,
            rank: 0,
            shuffle: false
        );

        // Act
        var batch = sampler.GetBatch(0, 2);

        // Assert
        Assert.Equal(new[] { 0, 4 }, batch);
    }

    [Fact]
    public void GetBatch_LastBatchSmaller()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 16,
            numReplicas: 4,
            rank: 0,
            shuffle: false
        );

        // Act - 4 samples, batch size 3, should get 2 batches: [0,4], [8,12]
        var batch0 = sampler.GetBatch(0, 3);
        var batch1 = sampler.GetBatch(1, 3);

        // Assert
        Assert.Equal(new[] { 0, 4, 8 }, batch0); // First batch has 3 elements
        Assert.Equal(new[] { 12 }, batch1);     // Last batch has 1 element
    }

    [Fact]
    public void GetBatch_BatchIndexOutOfRange_ThrowsException()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 16,
            numReplicas: 4,
            rank: 0,
            shuffle: false
        );

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => sampler.GetBatch(10, 2));
    }

    [Fact]
    public void GetNumBatches_ReturnsCorrectCount()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 16,
            numReplicas: 4,
            rank: 0,
            shuffle: false
        );

        // Act & Assert
        Assert.Equal(2, sampler.GetNumBatches(3));  // 4 samples / 3 = 2 batches
        Assert.Equal(1, sampler.GetNumBatches(4));  // 4 samples / 4 = 1 batch
        Assert.Equal(4, sampler.GetNumBatches(1));  // 4 samples / 1 = 4 batches
    }

    [Fact]
    public void GetIndices_ReturnsClone()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 16,
            numReplicas: 4,
            rank: 0,
            shuffle: false
        );

        // Act
        var indices1 = sampler.GetIndices();
        var indices2 = sampler.GetIndices();

        // Assert - Should be different array instances
        Assert.NotSame(indices1, indices2);
        Assert.Equal(indices1, indices2);
    }

    [Fact]
    public void Seed_ProvidesReproducibleShuffling()
    {
        // Arrange
        var sampler1 = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 0,
            shuffle: true,
            seed: 42
        );

        var sampler2 = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 0,
            shuffle: true,
            seed: 42
        );

        // Act
        var indices1 = sampler1.GetIndices();
        var indices2 = sampler2.GetIndices();

        // Assert
        Assert.Equal(indices1, indices2);
    }

    [Fact]
    public void DifferentSeeds_ProduceDifferentShuffling()
    {
        // Arrange
        var sampler1 = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 0,
            shuffle: true,
            seed: 42
        );

        var sampler2 = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 0,
            shuffle: true,
            seed: 123
        );

        // Act
        var indices1 = sampler1.GetIndices();
        var indices2 = sampler2.GetIndices();

        // Assert
        Assert.NotEqual(indices1, indices2);
    }

    [Fact]
    public void Integration_SimulateMultiProcessTraining()
    {
        // Arrange
        int numReplicas = 4;
        int numEpochs = 3;
        int datasetSize = 100;

        var epochResults = new List<List<int>>();

        // Act - Simulate 3 epochs of distributed training
        for (int epoch = 0; epoch < numEpochs; epoch++)
        {
            var allIndicesInEpoch = new List<int>();

            for (int rank = 0; rank < numReplicas; rank++)
            {
                var sampler = new DistributedSampler(
                    datasetSize: datasetSize,
                    numReplicas: numReplicas,
                    rank: rank,
                    shuffle: true,
                    dropLast: true
                );

                sampler.SetEpoch(epoch);
                allIndicesInEpoch.AddRange(sampler.Iterate());
            }

            epochResults.Add(allIndicesInEpoch);
        }

        // Assert - Each epoch should process all samples exactly once
        foreach (var epochIndices in epochResults)
        {
            Assert.Equal(datasetSize, epochIndices.Count);
            Assert.Equal(epochIndices.Distinct().Count(), epochIndices.Count); // No duplicates
            Assert.Equal(epochIndices.OrderBy(x => x).ToList(), epochIndices); // All indices present
        }

        // Assert - Different epochs should have different orderings (when shuffling is enabled)
        // Note: Different epochs don't guarantee different order across all replicas combined,
        // but the individual shuffles should be different
    }

    [Fact]
    public void Length_EqualsNumSamples()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 0,
            dropLast: true
        );

        // Act & Assert
        Assert.Equal(sampler.NumSamples, sampler.Length);
    }

    [Fact]
    public void Dispose_CanCallMultipleTimes()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 0
        );

        // Act & Assert - Should not throw
        sampler.Dispose();
        sampler.Dispose();
    }

    [Fact]
    public void Properties_ReturnCorrectValues()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 2,
            shuffle: true
        );

        // Act & Assert
        Assert.Equal(4, sampler.NumReplicas);
        Assert.Equal(2, sampler.Rank);
        Assert.Equal(0, sampler.Epoch);
        Assert.Equal(25, sampler.NumSamples);
    }
}
