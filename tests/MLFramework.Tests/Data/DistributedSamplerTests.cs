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
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DistributedSampler(
                datasetSize: -1,
                numReplicas: 4,
                rank: 0
            )
        );
    }

    [Fact]
    public void Constructor_ZeroDatasetSize_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DistributedSampler(
                datasetSize: 0,
                numReplicas: 4,
                rank: 0
            )
        );
    }

    [Fact]
    public void Constructor_ZeroNumReplicas_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
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
        Assert.Throws<ArgumentOutOfRangeException>(() =>
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
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DistributedSampler(
                datasetSize: 100,
                numReplicas: 4,
                rank: 4
            )
        );
    }

    [Fact]
    public void Constructor_SingleReplica_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new DistributedSampler(
                datasetSize: 100,
                numReplicas: 1,
                rank: 0
            )
        );
    }

    [Fact]
    public void CalculatePerReplicaSize_DropLastEvenlyDivisible_ReturnsEqualSize()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 0,
            dropLast: true
        );

        // Act & Assert
        Assert.Equal(25, sampler.Length);
    }

    [Fact]
    public void CalculatePerReplicaSize_DropLastNotEvenlyDivisible_DropsExtra()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 103,
            numReplicas: 4,
            rank: 0,
            dropLast: true
        );

        // Act & Assert
        Assert.Equal(25, sampler.Length); // 100 / 4 = 25, 3 samples dropped
    }

    [Fact]
    public void CalculatePerReplicaSize_NoDropLast_LowerReplicasGetsExtra()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 103,
            numReplicas: 4,
            rank: 3, // Last replica
            dropLast: false
        );

        // Act & Assert
        Assert.Equal(26, sampler.Length); // 103 / 4 = 25, +3 for last replica
    }

    [Fact]
    public void CalculatePerReplicaSize_NoDropLast_NonLastReplicaDoesntGetExtra()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 103,
            numReplicas: 4,
            rank: 0,
            dropLast: false
        );

        // Act & Assert
        Assert.Equal(25, sampler.Length);
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
        Assert.Equal(allIndices.OrderBy(x => x).ToList(), allIndices); // Sorted matches
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
    public void Iterate_SequentialIndicesWithinReplica()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 1,
            shuffle: false
        );

        // Act
        var indices = sampler.Iterate().ToList();

        // Assert
        Assert.Equal(new[] { 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49 }, indices);
    }

    [Fact]
    public void Iterate_SuffleTrue_DifferentOrderEachEpoch()
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
        var indices0 = sampler.Iterate().ToList();
        sampler.SetEpoch(1);
        var indices1 = sampler.Iterate().ToList();

        // Assert
        Assert.NotEqual(indices0, indices1);
        Assert.Equal(indices0.OrderBy(x => x).ToList(), indices0); // Same elements, different order
        Assert.Equal(indices1.OrderBy(x => x).ToList(), indices1);
    }

    [Fact]
    public void Iterate_SuffleFalse_SameOrderEachEpoch()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 0,
            shuffle: false
        );

        // Act
        var indices0 = sampler.Iterate().ToList();
        sampler.SetEpoch(1);
        var indices1 = sampler.Iterate().ToList();

        // Assert
        Assert.Equal(indices0, indices1);
    }

    [Fact]
    public void SetEpoch_ValidEpoch_UpdatesEpoch()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 0
        );

        // Act
        sampler.SetEpoch(5);

        // Assert
        Assert.Equal(5, sampler.Epoch);
    }

    [Fact]
    public void SetEpoch_NegativeEpoch_ThrowsException()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 0
        );

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => sampler.SetEpoch(-1));
    }

    [Fact]
    public void SetEpoch_ZeroEpoch_Accepted()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 0
        );

        // Act & Assert
        sampler.SetEpoch(0);
        Assert.Equal(0, sampler.Epoch);
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
        var indices1 = sampler1.Iterate().ToList();
        sampler1.SetEpoch(1);
        var indices1Epoch1 = sampler1.Iterate().ToList();

        var indices2 = sampler2.Iterate().ToList();
        sampler2.SetEpoch(1);
        var indices2Epoch1 = sampler2.Iterate().ToList();

        // Assert
        Assert.Equal(indices1, indices2);
        Assert.Equal(indices1Epoch1, indices2Epoch1);
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
        var indices1 = sampler1.Iterate().ToList();
        var indices2 = sampler2.Iterate().ToList();

        // Assert
        Assert.NotEqual(indices1, indices2);
    }

    [Fact]
    public void Length_ReturnsCorrectSizeForAllReplicas()
    {
        // Arrange
        var numReplicas = 4;
        var datasetSize = 103;
        var total = 0;

        // Act
        for (int rank = 0; rank < numReplicas; rank++)
        {
            var sampler = new DistributedSampler(
                datasetSize: datasetSize,
                numReplicas: numReplicas,
                rank: rank,
                dropLast: false
            );
            total += sampler.Length;
        }

        // Assert
        Assert.Equal(datasetSize, total);
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

        // Assert - Different epochs should have different orderings
        Assert.NotEqual(epochResults[0], epochResults[1]);
        Assert.NotEqual(epochResults[1], epochResults[2]);
    }

    [Fact]
    public void Iterate_MultipleCalls_ReturnsSameIndices()
    {
        // Arrange
        var sampler = new DistributedSampler(
            datasetSize: 100,
            numReplicas: 4,
            rank: 0,
            shuffle: false
        );

        // Act
        var first = sampler.Iterate().ToList();
        var second = sampler.Iterate().ToList();

        // Assert
        Assert.Equal(first, second);
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
    }
}
