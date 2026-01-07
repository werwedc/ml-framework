using MlFramework.Inference.PagedAttention.Sharing;
using Xunit;

namespace MLFramework.Tests.Inference.PagedAttention.Sharing;

public class BlockShareManagerTests
{
    [Fact]
    public void Constructor_InitializesCorrectly()
    {
        var manager = new BlockShareManager();

        var stats = manager.GetStats();
        Assert.Equal(0, stats.TotalSharedBlocks);
        Assert.Equal(0, stats.TotalBlocksReferenced);
        Assert.Equal(0, stats.TotalReferences);
        Assert.Equal(0, stats.AverageReferenceCount);
    }

    [Fact]
    public void ShareBlock_SingleSequence_SetsReferenceCountToOne()
    {
        var manager = new BlockShareManager();

        manager.ShareBlock(1, new List<int> { 100 });

        Assert.Equal(1, manager.GetReferenceCount(1));
        Assert.False(manager.IsBlockShared(1));
        Assert.Contains(100, manager.GetBlockUsers(1));
    }

    [Fact]
    public void ShareBlock_MultipleSequences_SharesBlock()
    {
        var manager = new BlockShareManager();

        manager.ShareBlock(1, new List<int> { 100, 101, 102 });

        Assert.Equal(3, manager.GetReferenceCount(1));
        Assert.True(manager.IsBlockShared(1));
        Assert.Equal(3, manager.GetBlockUsers(1).Count);
    }

    [Fact]
    public void ShareBlock_AdditionalSequences_IncrementsReferenceCount()
    {
        var manager = new BlockShareManager();

        // Share with two sequences
        manager.ShareBlock(1, new List<int> { 100, 101 });
        Assert.Equal(2, manager.GetReferenceCount(1));

        // Share with one more
        manager.ShareBlock(1, new List<int> { 102 });
        Assert.Equal(3, manager.GetReferenceCount(1));
    }

    [Fact]
    public void ReleaseSequence_RemovesReferences()
    {
        var manager = new BlockShareManager();

        manager.ShareBlock(1, new List<int> { 100, 101, 102 });
        Assert.Equal(3, manager.GetReferenceCount(1));

        var freed = manager.ReleaseSequence(100);
        Assert.Empty(freed);
        Assert.Equal(2, manager.GetReferenceCount(1));
    }

    [Fact]
    public void ReleaseSequence_LastSequence_FreesBlock()
    {
        var manager = new BlockShareManager();

        manager.ShareBlock(1, new List<int> { 100 });
        Assert.Equal(1, manager.GetReferenceCount(1));

        var freed = manager.ReleaseSequence(100);
        Assert.Single(freed);
        Assert.Contains(1, freed);
        Assert.Equal(0, manager.GetReferenceCount(1));
    }

    [Fact]
    public void ReleaseSequence_MultipleBlocks_FreesCorrectBlocks()
    {
        var manager = new BlockShareManager();

        manager.ShareBlock(1, new List<int> { 100 });
        manager.ShareBlock(2, new List<int> { 100, 101 });
        manager.ShareBlock(3, new List<int> { 101 });

        var freed = manager.ReleaseSequence(100);

        Assert.Contains(1, freed); // Block 1 was only used by 100
        Assert.DoesNotContain(2, freed); // Block 2 still used by 101
        Assert.Equal(0, manager.GetReferenceCount(1));
        Assert.Equal(1, manager.GetReferenceCount(2));
    }

    [Fact]
    public void IsBlockShared_UnsharedBlock_ReturnsFalse()
    {
        var manager = new BlockShareManager();

        manager.ShareBlock(1, new List<int> { 100 });

        Assert.False(manager.IsBlockShared(1));
    }

    [Fact]
    public void IsBlockShared_SharedBlock_ReturnsTrue()
    {
        var manager = new BlockShareManager();

        manager.ShareBlock(1, new List<int> { 100, 101 });

        Assert.True(manager.IsBlockShared(1));
    }

    [Fact]
    public void GetReferenceCount_NonExistentBlock_ReturnsZero()
    {
        var manager = new BlockShareManager();

        Assert.Equal(0, manager.GetReferenceCount(999));
    }

    [Fact]
    public void GetSequenceBlocks_ReturnsCorrectBlocks()
    {
        var manager = new BlockShareManager();

        manager.ShareBlock(1, new List<int> { 100 });
        manager.ShareBlock(2, new List<int> { 100 });
        manager.ShareBlock(3, new List<int> { 101 });

        var blocksFor100 = manager.GetSequenceBlocks(100);
        Assert.Equal(2, blocksFor100.Count);
        Assert.Contains(1, blocksFor100);
        Assert.Contains(2, blocksFor100);

        var blocksFor101 = manager.GetSequenceBlocks(101);
        Assert.Single(blocksFor101);
        Assert.Contains(3, blocksFor101);
    }

    [Fact]
    public void GetBlockUsers_ReturnsCorrectSequences()
    {
        var manager = new BlockShareManager();

        manager.ShareBlock(1, new List<int> { 100, 101, 102 });

        var users = manager.GetBlockUsers(1);
        Assert.Equal(3, users.Count);
        Assert.Contains(100, users);
        Assert.Contains(101, users);
        Assert.Contains(102, users);
    }

    [Fact]
    public void GetStats_CalculatesCorrectly()
    {
        var manager = new BlockShareManager();

        manager.ShareBlock(1, new List<int> { 100, 101 });
        manager.ShareBlock(2, new List<int> { 100 });
        manager.ShareBlock(3, new List<int> { 101 });

        var stats = manager.GetStats();

        Assert.Equal(3, stats.TotalBlocksReferenced);
        Assert.Equal(4, stats.TotalReferences);
        Assert.Equal(1, stats.TotalSharedBlocks); // Only block 1 is shared
        Assert.Equal(4.0 / 3.0, stats.AverageReferenceCount);
    }

    [Fact]
    public void MultipleBlocks_IndependentSharing()
    {
        var manager = new BlockShareManager();

        manager.ShareBlock(1, new List<int> { 100, 101 });
        manager.ShareBlock(2, new List<int> { 102 });
        manager.ShareBlock(3, new List<int> { 102, 103 });

        Assert.Equal(2, manager.GetReferenceCount(1));
        Assert.Equal(1, manager.GetReferenceCount(2));
        Assert.Equal(2, manager.GetReferenceCount(3));

        Assert.True(manager.IsBlockShared(1));
        Assert.False(manager.IsBlockShared(2));
        Assert.True(manager.IsBlockShared(3));
    }

    [Fact]
    public void ReleaseNonExistentSequence_ReturnsEmptyList()
    {
        var manager = new BlockShareManager();

        var freed = manager.ReleaseSequence(999);

        Assert.Empty(freed);
    }
}
