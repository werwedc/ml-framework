using MlFramework.Inference.PagedAttention.Sharing;
using Xunit;

namespace MLFramework.Tests.Inference.PagedAttention.Sharing;

public class BeamSearchBlockSharingTests
{
    [Fact]
    public void InitializeBeams_CreatesCorrectNumberOfBeams()
    {
        var shareManager = new BlockShareManager();
        var beamSharing = new BeamSearchBlockSharing(shareManager, beamWidth: 4);

        var beamIds = beamSharing.InitializeBeams(baseSequenceId: 1, prefixLength: 10);

        Assert.Equal(4, beamIds.Count);
        Assert.Contains(1000, beamIds); // 1 * 1000 + 0
        Assert.Contains(1001, beamIds); // 1 * 1000 + 1
        Assert.Contains(1002, beamIds); // 1 * 1000 + 2
        Assert.Contains(1003, beamIds); // 1 * 1000 + 3
    }

    [Fact]
    public void InitializeBeams_CreatesBeamInfoForAllBeams()
    {
        var shareManager = new BlockShareManager();
        var beamSharing = new BeamSearchBlockSharing(shareManager, beamWidth: 3);

        var beamIds = beamSharing.InitializeBeams(baseSequenceId: 2, prefixLength: 20);

        foreach (var beamId in beamIds)
        {
            var info = beamSharing.GetBeamInfo(beamId);
            Assert.NotNull(info);
            Assert.Equal(2, info.BaseSequenceId);
            Assert.Equal(20, info.DivergencePoint);
        }
    }

    [Fact]
    public void InitializeBeams_SetsBeamIndexCorrectly()
    {
        var shareManager = new BlockShareManager();
        var beamSharing = new BeamSearchBlockSharing(shareManager, beamWidth: 5);

        var beamIds = beamSharing.InitializeBeams(baseSequenceId: 1, prefixLength: 5);

        for (int i = 0; i < beamIds.Count; i++)
        {
            var info = beamSharing.GetBeamInfo(beamIds[i]);
            Assert.Equal(i, info.BeamIndex);
        }
    }

    [Fact]
    public void OnBeamDivergence_UpdatesDivergencePoint()
    {
        var shareManager = new BlockShareManager();
        var beamSharing = new BeamSearchBlockSharing(shareManager, beamWidth: 2);

        var beamIds = beamSharing.InitializeBeams(baseSequenceId: 1, prefixLength: 10);

        beamSharing.OnBeamDivergence(beamIds[0], divergencePoint: 15);

        var info = beamSharing.GetBeamInfo(beamIds[0]);
        Assert.Equal(15, info.DivergencePoint);
    }

    [Fact]
    public void CleanupBeams_ReleasesAllBeams()
    {
        var shareManager = new BlockShareManager();
        var beamSharing = new BeamSearchBlockSharing(shareManager, beamWidth: 3);

        var beamIds = beamSharing.InitializeBeams(baseSequenceId: 1, prefixLength: 10);

        // Share some blocks with beams
        shareManager.ShareBlock(1, beamIds);

        var freedBlocks = beamSharing.CleanupBeams(beamIds);

        // Verify all beams are cleaned up
        foreach (var beamId in beamIds)
        {
            Assert.Null(beamSharing.GetBeamInfo(beamId));
        }
    }

    [Fact]
    public void CleanupBeams_ReturnsFreedBlocks()
    {
        var shareManager = new BlockShareManager();
        var beamSharing = new BeamSearchBlockSharing(shareManager, beamWidth: 2);

        var beamIds = beamSharing.InitializeBeams(baseSequenceId: 1, prefixLength: 10);

        // Share blocks with beams
        shareManager.ShareBlock(1, beamIds);
        shareManager.ShareBlock(2, new List<int> { beamIds[0] }); // Unique block for first beam

        var freedBlocks = beamSharing.CleanupBeams(beamIds);

        // Block 1 should be freed (no references left)
        // Block 2 should be freed (only used by beams that were cleaned up)
        Assert.Contains(1, freedBlocks);
        Assert.Contains(2, freedBlocks);
    }

    [Fact]
    public void GetBeamInfo_NonExistentBeam_ReturnsNull()
    {
        var shareManager = new BlockShareManager();
        var beamSharing = new BeamSearchBlockSharing(shareManager, beamWidth: 2);

        Assert.Null(beamSharing.GetBeamInfo(999));
    }

    [Fact]
    public void MultipleInitializations_IndependentBeamSets()
    {
        var shareManager = new BlockShareManager();
        var beamSharing = new BeamSearchBlockSharing(shareManager, beamWidth: 2);

        var beamIds1 = beamSharing.InitializeBeams(baseSequenceId: 1, prefixLength: 10);
        var beamIds2 = beamSharing.InitializeBeams(baseSequenceId: 2, prefixLength: 15);

        // Verify beam info is correct for each set
        foreach (var beamId in beamIds1)
        {
            var info = beamSharing.GetBeamInfo(beamId);
            Assert.Equal(1, info.BaseSequenceId);
            Assert.Equal(10, info.DivergencePoint);
        }

        foreach (var beamId in beamIds2)
        {
            var info = beamSharing.GetBeamInfo(beamId);
            Assert.Equal(2, info.BaseSequenceId);
            Assert.Equal(15, info.DivergencePoint);
        }
    }

    [Fact]
    public void SelectiveCleanup_KeepsSomeBeams()
    {
        var shareManager = new BlockShareManager();
        var beamSharing = new BeamSearchBlockSharing(shareManager, beamWidth: 4);

        var beamIds = beamSharing.InitializeBeams(baseSequenceId: 1, prefixLength: 10);

        // Share blocks
        shareManager.ShareBlock(1, beamIds);
        shareManager.ShareBlock(2, new List<int> { beamIds[0], beamIds[1] });

        // Cleanup only half the beams
        var beamsToCleanup = beamIds.Take(2).ToList();
        var freedBlocks = beamSharing.CleanupBeams(beamsToCleanup);

        // Verify remaining beams still exist
        Assert.NotNull(beamSharing.GetBeamInfo(beamIds[2]));
        Assert.NotNull(beamSharing.GetBeamInfo(beamIds[3]));

        // Verify cleaned up beams are gone
        Assert.Null(beamSharing.GetBeamInfo(beamIds[0]));
        Assert.Null(beamSharing.GetBeamInfo(beamIds[1]));
    }
}
