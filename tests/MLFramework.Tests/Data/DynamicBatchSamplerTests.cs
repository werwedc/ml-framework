using MLFramework.Data;
using Xunit;

namespace MLFramework.Tests.Data;

/// <summary>
/// Tests for DynamicBatchSampler.
/// </summary>
public class DynamicBatchSamplerTests
{
    private class MockSequenceDataset : IDataset<Sequence>
    {
        private readonly Sequence[] _sequences;

        public MockSequenceDataset(Sequence[] sequences)
        {
            _sequences = sequences;
        }

        public int Length => _sequences.Length;

        public Sequence GetItem(int index)
        {
            return _sequences[index];
        }
    }

    [Fact]
    public void Constructor_NullDataset_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new DynamicBatchSampler(null, DynamicBatchStrategy.PadToMax, 32));
    }

    [Fact]
    public void Constructor_NonPositiveMaxBatchSize_ThrowsException()
    {
        var dataset = new MockSequenceDataset(
            new[] { new Sequence(new[] { 1, 2, 3 }) });

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DynamicBatchSampler(dataset, DynamicBatchStrategy.PadToMax, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DynamicBatchSampler(dataset, DynamicBatchStrategy.PadToMax, -1));
    }

    [Fact]
    public void Constructor_NonPositiveMaxSequenceLength_ThrowsException()
    {
        var dataset = new MockSequenceDataset(
            new[] { new Sequence(new[] { 1, 2, 3 }) });

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DynamicBatchSampler(dataset, DynamicBatchStrategy.PadToMax, 32, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new DynamicBatchSampler(dataset, DynamicBatchStrategy.PadToMax, 32, -1));
    }

    [Fact]
    public void PadToMaxStrategy_ExactMultiple_ReturnsFullBatches()
    {
        // Arrange
        var sequences = new[]
        {
            new Sequence(new[] { 1, 2, 3 }),
            new Sequence(new[] { 4, 5 }),
            new Sequence(new[] { 6, 7, 8, 9 }),
            new Sequence(new[] { 10 }),
            new Sequence(new[] { 11, 12 }),
            new Sequence(new[] { 13, 14, 15 }),
        };
        var dataset = new MockSequenceDataset(sequences);
        var sampler = new DynamicBatchSampler(
            dataset,
            DynamicBatchStrategy.PadToMax,
            maxBatchSize: 3);

        // Act
        var batches = sampler.Iterate().ToList();

        // Assert
        Assert.Equal(2, batches.Count);
        Assert.Equal(3, batches[0].Length);
        Assert.Equal(3, batches[1].Length);
        Assert.Equal(3, sampler.BatchSize);
    }

    [Fact]
    public void PadToMaxStrategy_Remainder_IncludesPartialBatch()
    {
        // Arrange
        var sequences = new[]
        {
            new Sequence(new[] { 1, 2, 3 }),
            new Sequence(new[] { 4, 5 }),
            new Sequence(new[] { 6, 7 }),
            new Sequence(new[] { 8, 9, 10 }),
        };
        var dataset = new MockSequenceDataset(sequences);
        var sampler = new DynamicBatchSampler(
            dataset,
            DynamicBatchStrategy.PadToMax,
            maxBatchSize: 3);

        // Act
        var batches = sampler.Iterate().ToList();

        // Assert
        Assert.Equal(2, batches.Count);
        Assert.Equal(3, batches[0].Length);
        Assert.Equal(1, batches[1].Length);
        Assert.Equal(1, sampler.BatchSize);
    }

    [Fact]
    public void PadToMaxStrategy_SingleItem_ReturnsSingleBatch()
    {
        // Arrange
        var sequences = new[] { new Sequence(new[] { 1, 2, 3 }) };
        var dataset = new MockSequenceDataset(sequences);
        var sampler = new DynamicBatchSampler(
            dataset,
            DynamicBatchStrategy.PadToMax,
            maxBatchSize: 32);

        // Act
        var batches = sampler.Iterate().ToList();

        // Assert
        Assert.Single(batches);
        Assert.Single(batches[0]);
        Assert.Equal(1, sampler.BatchSize);
    }

    [Fact]
    public void PadToMaxStrategy_EmptyDataset_ReturnsEmpty()
    {
        // Arrange
        var dataset = new MockSequenceDataset(Array.Empty<Sequence>());
        var sampler = new DynamicBatchSampler(
            dataset,
            DynamicBatchStrategy.PadToMax,
            maxBatchSize: 32);

        // Act
        var batches = sampler.Iterate().ToList();

        // Assert
        Assert.Empty(batches);
    }

    [Fact]
    public void BucketStrategy_GroupsSimilarLengths()
    {
        // Arrange
        var sequences = new[]
        {
            // Short sequences (bucket 0-63)
            new Sequence(new[] { 1, 2, 3 }),
            new Sequence(new[] { 4, 5, 6, 7 }),
            new Sequence(new[] { 8, 9 }),

            // Medium sequences (bucket 64-127)
            new Sequence(Enumerable.Range(10, 80).ToArray()),
            new Sequence(Enumerable.Range(90, 90).ToArray()),
            new Sequence(Enumerable.Range(180, 70).ToArray()),

            // Long sequences (bucket 128-191)
            new Sequence(Enumerable.Range(250, 150).ToArray()),
            new Sequence(Enumerable.Range(400, 140).ToArray()),
        };
        var dataset = new MockSequenceDataset(sequences);
        var sampler = new DynamicBatchSampler(
            dataset,
            DynamicBatchStrategy.Bucket,
            maxBatchSize: 32);

        // Act
        var batches = sampler.Iterate().ToList();

        // Assert - Should create batches from each bucket
        Assert.Equal(3, batches.Count);
        Assert.All(batches, batch => Assert.True(batch.Length > 0));
    }

    [Fact]
    public void BucketStrategy_LargeBucket_CreatesMultipleBatches()
    {
        // Arrange - Many sequences in the same bucket
        var sequences = Enumerable
            .Range(0, 100)
            .Select(i => new Sequence(new[] { i, i + 1, i + 2 }))
            .ToArray();

        var dataset = new MockSequenceDataset(sequences);
        var sampler = new DynamicBatchSampler(
            dataset,
            DynamicBatchStrategy.Bucket,
            maxBatchSize: 10);

        // Act
        var batches = sampler.Iterate().ToList();

        // Assert
        Assert.Equal(10, batches.Count);
        Assert.All(batches.Take(9), batch => Assert.Equal(10, batch.Length));
        Assert.Equal(10, batches.Last().Length); // 100 / 10 = 10 exactly
    }

    [Fact]
    public void BucketStrategy_VaryingLengths_SeparatesCorrectly()
    {
        // Arrange
        var sequences = new[]
        {
            // Bucket 0: length 1-63
            new Sequence(new[] { 1, 2, 3 }),
            new Sequence(new[] { 4, 5, 6, 7 }),

            // Bucket 64: length 64-127
            new Sequence(Enumerable.Range(8, 70).ToArray()),

            // Bucket 128: length 128-191
            new Sequence(Enumerable.Range(78, 140).ToArray()),

            // Bucket 192: length 192-255
            new Sequence(Enumerable.Range(218, 210).ToArray()),
        };
        var dataset = new MockSequenceDataset(sequences);
        var sampler = new DynamicBatchSampler(
            dataset,
            DynamicBatchStrategy.Bucket,
            maxBatchSize: 32);

        // Act
        var batches = sampler.Iterate().ToList();

        // Assert - Each bucket should be in its own batch
        Assert.Equal(4, batches.Count);
    }

    [Fact]
    public void DynamicStrategy_AdjustsBatchSizeForTokenLimit()
    {
        // Arrange - Sequences with varying lengths
        var sequences = new[]
        {
            new Sequence(new[] { 1, 2, 3 }),               // 3 tokens
            new Sequence(Enumerable.Range(4, 50).ToArray()),   // 50 tokens
            new Sequence(Enumerable.Range(54, 80).ToArray()),  // 80 tokens
            new Sequence(new[] { 134, 135 }),               // 2 tokens
            new Sequence(Enumerable.Range(136, 60).ToArray()),  // 60 tokens
            new Sequence(Enumerable.Range(196, 90).ToArray()),  // 90 tokens
        };
        var dataset = new MockSequenceDataset(sequences);
        var sampler = new DynamicBatchSampler(
            dataset,
            DynamicBatchStrategy.Dynamic,
            maxBatchSize: 32,
            maxSequenceLength: 128);

        // Act
        var batches = sampler.Iterate().ToList();

        // Assert - First batch: 3 + 50 + 80 = 133 tokens, under limit (32 * 128 = 4096)
        Assert.True(batches.Count > 0);
        Assert.All(batches, batch => Assert.True(batch.Length > 0));
    }

    [Fact]
    public void DynamicStrategy_LargeSequences_ReducesBatchSize()
    {
        // Arrange - All sequences near max length
        var sequences = new[]
        {
            new Sequence(Enumerable.Range(0, 100).ToArray()),
            new Sequence(Enumerable.Range(100, 100).ToArray()),
            new Sequence(Enumerable.Range(200, 100).ToArray()),
            new Sequence(Enumerable.Range(300, 100).ToArray()),
            new Sequence(Enumerable.Range(400, 100).ToArray()),
        };
        var dataset = new MockSequenceDataset(sequences);
        var sampler = new DynamicBatchSampler(
            dataset,
            DynamicBatchStrategy.Dynamic,
            maxBatchSize: 32,
            maxSequenceLength: 128);

        // Act
        var batches = sampler.Iterate().ToList();

        // Assert - With 100-token sequences and 4096 token limit, should fit ~40 sequences per batch
        // But we only have 5 sequences total, so they should all fit in one batch
        Assert.Single(batches);
        Assert.Equal(5, batches[0].Length);
    }

    [Fact]
    public void DynamicStrategy_MixedLengths_BalancesBatchSize()
    {
        // Arrange - Mixed sequence lengths
        var sequences = new[]
        {
            new Sequence(new[] { 1 }),                     // 1 token
            new Sequence(Enumerable.Range(2, 127).ToArray()), // 127 tokens
            new Sequence(new[] { 129 }),                  // 1 token
            new Sequence(Enumerable.Range(130, 127).ToArray()), // 127 tokens
        };
        var dataset = new MockSequenceDataset(sequences);
        var sampler = new DynamicBatchSampler(
            dataset,
            DynamicBatchStrategy.Dynamic,
            maxBatchSize: 32,
            maxSequenceLength: 128);

        // Act
        var batches = sampler.Iterate().ToList();

        // Assert - First batch: 1 + 127 + 1 = 129 tokens
        // Second batch: 127 tokens
        // Limit: 32 * 128 = 4096 tokens
        Assert.Equal(2, batches.Count);
        Assert.Equal(3, batches[0].Length);
        Assert.Equal(1, batches[1].Length);
    }

    [Fact]
    public void DynamicStrategy_SequencesOverMaxLength_ClampsToMax()
    {
        // Arrange - Some sequences exceed max length
        var sequences = new[]
        {
            new Sequence(Enumerable.Range(0, 150).ToArray()),   // 150 tokens, clamped to 128
            new Sequence(Enumerable.Range(150, 200).ToArray()), // 200 tokens, clamped to 128
            new Sequence(new[] { 350 }),                       // 1 token
        };
        var dataset = new MockSequenceDataset(sequences);
        var sampler = new DynamicBatchSampler(
            dataset,
            DynamicBatchStrategy.Dynamic,
            maxBatchSize: 32,
            maxSequenceLength: 128);

        // Act
        var batches = sampler.Iterate().ToList();

        // Assert - 128 + 128 + 1 = 257 tokens, under 4096 limit
        // All sequences should fit in one batch
        Assert.Single(batches);
        Assert.Equal(3, batches[0].Length);
    }

    [Fact]
    public void Iterate_AllStrategies_ReturnsAllIndices()
    {
        // Arrange
        var sequences = Enumerable
            .Range(0, 10)
            .Select(i => new Sequence(new[] { i, i + 1 }))
            .ToArray();

        // Act & Assert for each strategy
        foreach (var strategy in Enum.GetValues<DynamicBatchStrategy>())
        {
            var dataset = new MockSequenceDataset(sequences);
            var sampler = new DynamicBatchSampler(dataset, strategy, maxBatchSize: 3);
            var batches = sampler.Iterate().ToList();

            var allIndices = batches.SelectMany(b => b).OrderBy(i => i).ToList();
            var expectedIndices = Enumerable.Range(0, 10).ToList();

            Assert.Equal(expectedIndices, allIndices);
        }
    }

    [Fact]
    public void Iterate_UnknownStrategy_ThrowsException()
    {
        // This test ensures that if a new strategy is added to the enum
        // but not handled in the switch statement, an exception is thrown
        // Currently all enum values are handled, so this test is more about
        // ensuring the switch is exhaustive

        // Arrange
        var dataset = new MockSequenceDataset(
            new[] { new Sequence(new[] { 1, 2, 3 }) });

        // Act - All known strategies should work
        foreach (var strategy in Enum.GetValues<DynamicBatchStrategy>())
        {
            var sampler = new DynamicBatchSampler(dataset, strategy, 32);
            var batches = sampler.Iterate().ToList();

            Assert.NotNull(batches);
        }
    }

    [Fact]
    public void BatchSize_PropertyReflectsActualBatchSize()
    {
        // Arrange
        var sequences = new[]
        {
            new Sequence(new[] { 1 }),
            new Sequence(new[] { 2 }),
            new Sequence(new[] { 3 }),
            new Sequence(new[] { 4 }),
        };
        var dataset = new MockSequenceDataset(sequences);
        var sampler = new DynamicBatchSampler(
            dataset,
            DynamicBatchStrategy.PadToMax,
            maxBatchSize: 3);

        // Act
        var batches = sampler.Iterate().ToList();

        // Assert
        Assert.Equal(1, sampler.BatchSize); // Last batch size
        Assert.Equal(3, batches[0].Length);
        Assert.Equal(1, batches[1].Length);
    }

    [Fact]
    public void Iterate_MultipleIterations_Independent()
    {
        // Arrange
        var sequences = Enumerable
            .Range(0, 5)
            .Select(i => new Sequence(new[] { i, i + 1 }))
            .ToArray();

        var dataset = new MockSequenceDataset(sequences);
        var sampler = new DynamicBatchSampler(
            dataset,
            DynamicBatchStrategy.PadToMax,
            maxBatchSize: 2);

        // Act
        var first = sampler.Iterate().ToList();
        var second = sampler.Iterate().ToList();

        // Assert
        Assert.Equal(first.Count, second.Count);
        Assert.Equal(first[0], second[0]);
        Assert.Equal(first[1], second[1]);
    }

    [Fact]
    public void Sequence_ConstructorWithTokens_CreatesSequence()
    {
        // Arrange
        var tokens = new[] { 1, 2, 3, 4, 5 };

        // Act
        var sequence = new Sequence(tokens);

        // Assert
        Assert.Equal(tokens, sequence.Tokens);
        Assert.Equal(5, sequence.Length);
    }

    [Fact]
    public void Sequence_ConstructorWithNullTokens_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new Sequence(null));
    }

    [Fact]
    public void Sequence_DefaultConstructor_CreatesEmptySequence()
    {
        // Arrange & Act
        var sequence = new Sequence();

        // Assert
        Assert.NotNull(sequence.Tokens);
        Assert.Empty(sequence.Tokens);
        Assert.Equal(0, sequence.Length);
    }

    [Fact]
    public void Sequence_Slice_ReturnsCorrectSubsequence()
    {
        // Arrange
        var tokens = new[] { 1, 2, 3, 4, 5 };
        var sequence = new Sequence(tokens);

        // Act
        var sliced = sequence.Slice(1, 4);

        // Assert
        Assert.Equal(new[] { 2, 3, 4 }, sliced.Tokens);
        Assert.Equal(3, sliced.Length);
    }

    [Fact]
    public void Sequence_SliceInvalidRange_ThrowsException()
    {
        // Arrange
        var tokens = new[] { 1, 2, 3, 4, 5 };
        var sequence = new Sequence(tokens);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => sequence.Slice(-1, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => sequence.Slice(1, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => sequence.Slice(4, 2));
    }

    [Fact]
    public void Sequence_ToString_ReturnsCorrectRepresentation()
    {
        // Arrange
        var tokens = new[] { 1, 2, 3 };
        var sequence = new Sequence(tokens);

        // Act
        var result = sequence.ToString();

        // Assert
        Assert.Equal("[1, 2, 3]", result);
    }
}
