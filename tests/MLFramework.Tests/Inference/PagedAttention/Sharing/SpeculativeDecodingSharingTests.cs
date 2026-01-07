using MlFramework.Inference.PagedAttention.Sharing;
using Xunit;

namespace MLFramework.Tests.Inference.PagedAttention.Sharing;

public class SpeculativeDecodingSharingTests
{
    [Fact]
    public void Constructor_InitializesCorrectly()
    {
        var shareManager = new BlockShareManager();
        var specSharing = new SpeculativeDecodingSharing(shareManager, speculationLength: 4);

        var state = specSharing.GetSpeculativeState(1);
        Assert.Null(state);
    }

    [Fact]
    public void AllocateSpeculativeBlocks_CreatesState()
    {
        var shareManager = new BlockShareManager();
        var specSharing = new SpeculativeDecodingSharing(shareManager, speculationLength: 4);

        var blockIds = new List<int> { 10, 11, 12 };
        specSharing.AllocateSpeculativeBlocks(
            sequenceId: 1,
            baseSequenceLength: 100,
            speculativeBlockIds: blockIds
        );

        var state = specSharing.GetSpeculativeState(1);
        Assert.NotNull(state);
        Assert.Equal(100, state.BaseSequenceLength);
        Assert.Equal(blockIds, state.SpeculativeBlockIds);
        Assert.Equal(0, state.VerifiedCount);
    }

    [Fact]
    public void AllocateSpeculativeBlocks_StoresBlockIds()
    {
        var shareManager = new BlockShareManager();
        var specSharing = new SpeculativeDecodingSharing(shareManager, speculationLength: 4);

        var blockIds = new List<int> { 20, 21, 22, 23 };
        specSharing.AllocateSpeculativeBlocks(
            sequenceId: 2,
            baseSequenceLength: 50,
            speculativeBlockIds: blockIds
        );

        var state = specSharing.GetSpeculativeState(2);
        Assert.Equal(4, state.SpeculativeBlockIds.Count);
        Assert.Equal(20, state.SpeculativeBlockIds[0]);
        Assert.Equal(21, state.SpeculativeBlockIds[1]);
        Assert.Equal(22, state.SpeculativeBlockIds[2]);
        Assert.Equal(23, state.SpeculativeBlockIds[3]);
    }

    [Fact]
    public void VerifySpeculation_AllTokensVerified_KeepsAllBlocks()
    {
        var shareManager = new BlockShareManager();
        var specSharing = new SpeculativeDecodingSharing(shareManager, speculationLength: 4);

        var blockIds = new List<int> { 10, 11, 12 };
        specSharing.AllocateSpeculativeBlocks(
            sequenceId: 1,
            baseSequenceLength: 100,
            speculativeBlockIds: blockIds
        );

        var (keptBlocks, freedBlocks) = specSharing.VerifySpeculation(sequenceId: 1, verifiedTokens: 3);

        Assert.Equal(3, keptBlocks.Count);
        Assert.Contains(10, keptBlocks);
        Assert.Contains(11, keptBlocks);
        Assert.Contains(12, keptBlocks);
        Assert.Empty(freedBlocks);
    }

    [Fact]
    public void VerifySpeculation_PartialVerification_KeepsAndFreesCorrectly()
    {
        var shareManager = new BlockShareManager();
        var specSharing = new SpeculativeDecodingSharing(shareManager, speculationLength: 4);

        var blockIds = new List<int> { 10, 11, 12, 13 };
        specSharing.AllocateSpeculativeBlocks(
            sequenceId: 1,
            baseSequenceLength: 100,
            speculativeBlockIds: blockIds
        );

        var (keptBlocks, freedBlocks) = specSharing.VerifySpeculation(sequenceId: 1, verifiedTokens: 2);

        Assert.Equal(2, keptBlocks.Count);
        Assert.Contains(10, keptBlocks);
        Assert.Contains(11, keptBlocks);

        Assert.Equal(2, freedBlocks.Count);
        Assert.Contains(12, freedBlocks);
        Assert.Contains(13, freedBlocks);
    }

    [Fact]
    public void VerifySpeculation_NoTokensVerified_FreesAllBlocks()
    {
        var shareManager = new BlockShareManager();
        var specSharing = new SpeculativeDecodingSharing(shareManager, speculationLength: 4);

        var blockIds = new List<int> { 10, 11, 12 };
        specSharing.AllocateSpeculativeBlocks(
            sequenceId: 1,
            baseSequenceLength: 100,
            speculativeBlockIds: blockIds
        );

        var (keptBlocks, freedBlocks) = specSharing.VerifySpeculation(sequenceId: 1, verifiedTokens: 0);

        Assert.Empty(keptBlocks);
        Assert.Equal(3, freedBlocks.Count);
        Assert.Contains(10, freedBlocks);
        Assert.Contains(11, freedBlocks);
        Assert.Contains(12, freedBlocks);
    }

    [Fact]
    public void VerifySpeculation_UpdatesVerifiedCount()
    {
        var shareManager = new BlockShareManager();
        var specSharing = new SpeculativeDecodingSharing(shareManager, speculationLength: 4);

        var blockIds = new List<int> { 10, 11, 12, 13 };
        specSharing.AllocateSpeculativeBlocks(
            sequenceId: 1,
            baseSequenceLength: 100,
            speculativeBlockIds: blockIds
        );

        specSharing.VerifySpeculation(sequenceId: 1, verifiedTokens: 3);

        var state = specSharing.GetSpeculativeState(1);
        Assert.NotNull(state);
        Assert.Equal(3, state.VerifiedCount);
    }

    [Fact]
    public void RejectAllSpeculation_FreesAllBlocks()
    {
        var shareManager = new BlockShareManager();
        var specSharing = new SpeculativeDecodingSharing(shareManager, speculationLength: 4);

        var blockIds = new List<int> { 10, 11, 12 };
        specSharing.AllocateSpeculativeBlocks(
            sequenceId: 1,
            baseSequenceLength: 100,
            speculativeBlockIds: blockIds
        );

        var freedBlocks = specSharing.RejectAllSpeculation(sequenceId: 1);

        Assert.Equal(3, freedBlocks.Count);
        Assert.Contains(10, freedBlocks);
        Assert.Contains(11, freedBlocks);
        Assert.Contains(12, freedBlocks);
    }

    [Fact]
    public void RejectAllSpeculation_RemovesState()
    {
        var shareManager = new BlockShareManager();
        var specSharing = new SpeculativeDecodingSharing(shareManager, speculationLength: 4);

        var blockIds = new List<int> { 10, 11, 12 };
        specSharing.AllocateSpeculativeBlocks(
            sequenceId: 1,
            baseSequenceLength: 100,
            speculativeBlockIds: blockIds
        );

        specSharing.RejectAllSpeculation(sequenceId: 1);

        var state = specSharing.GetSpeculativeState(1);
        Assert.Null(state);
    }

    [Fact]
    public void RejectNonExistentSpeculation_ReturnsEmptyList()
    {
        var shareManager = new BlockShareManager();
        var specSharing = new SpeculativeDecodingSharing(shareManager, speculationLength: 4);

        var freedBlocks = specSharing.RejectAllSpeculation(sequenceId: 999);

        Assert.Empty(freedBlocks);
    }

    [Fact]
    public void MultipleSequences_IndependentSpeculativeStates()
    {
        var shareManager = new BlockShareManager();
        var specSharing = new SpeculativeDecodingSharing(shareManager, speculationLength: 4);

        // Allocate for sequence 1
        var blockIds1 = new List<int> { 10, 11 };
        specSharing.AllocateSpeculativeBlocks(
            sequenceId: 1,
            baseSequenceLength: 100,
            speculativeBlockIds: blockIds1
        );

        // Allocate for sequence 2
        var blockIds2 = new List<int> { 20, 21 };
        specSharing.AllocateSpeculativeBlocks(
            sequenceId: 2,
            baseSequenceLength: 200,
            speculativeBlockIds: blockIds2
        );

        // Verify sequence 1
        var (kept1, freed1) = specSharing.VerifySpeculation(sequenceId: 1, verifiedTokens: 1);
        Assert.Single(kept1);

        // Verify sequence 2 independently
        var (kept2, freed2) = specSharing.VerifySpeculation(sequenceId: 2, verifiedTokens: 2);
        Assert.Equal(2, kept2.Count);

        // Check states are independent
        var state1 = specSharing.GetSpeculativeState(1);
        var state2 = specSharing.GetSpeculativeState(2);
        Assert.Equal(1, state1.VerifiedCount);
        Assert.Equal(2, state2.VerifiedCount);
    }

    [Fact]
    public void GetSpeculativeState_NonExistentSequence_ReturnsNull()
    {
        var shareManager = new BlockShareManager();
        var specSharing = new SpeculativeDecodingSharing(shareManager, speculationLength: 4);

        Assert.Null(specSharing.GetSpeculativeState(999));
    }

    [Fact]
    public void CustomSpeculationLength_CreatesWithCorrectLength()
    {
        var shareManager = new BlockShareManager();
        var specSharing = new SpeculativeDecodingSharing(shareManager, speculationLength: 8);

        var blockIds = new List<int>();
        for (int i = 0; i < 8; i++)
        {
            blockIds.Add(10 + i);
        }

        specSharing.AllocateSpeculativeBlocks(
            sequenceId: 1,
            baseSequenceLength: 100,
            speculativeBlockIds: blockIds
        );

        var state = specSharing.GetSpeculativeState(1);
        Assert.Equal(8, state.SpeculativeBlockIds.Count);
    }
}
