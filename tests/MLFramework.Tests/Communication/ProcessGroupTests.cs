namespace MLFramework.Communication.Tests;

using MLFramework.Communication;
using System;
using System.Linq;
using Xunit;

public class ProcessGroupLiteTests
{
    [Fact]
    public void Constructor_ValidParameters_CreatesProcessGroup()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var ranks = new[] { 0, 1, 2 };
        var group = new ProcessGroupLite(backend, ranks, myRank: 0, groupName: "test");

        Assert.Equal("test", group.GroupName);
        Assert.Equal(0, group.Rank);
        Assert.Equal(3, group.GroupSize);
        Assert.True(group.ContainsRank(0));
        Assert.True(group.ContainsRank(1));
        Assert.True(group.ContainsRank(2));
        Assert.False(group.ContainsRank(3));
    }

    [Fact]
    public void Constructor_NullBackend_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new ProcessGroupLite(null, new[] { 0, 1, 2 }, myRank: 0, groupName: "test"));
    }

    [Fact]
    public void Constructor_NullGroupName_ThrowsArgumentNullException()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        Assert.Throws<ArgumentNullException>(() =>
            new ProcessGroupLite(backend, new[] { 0, 1, 2 }, myRank: 0, groupName: null));
    }

    [Fact]
    public void Constructor_NullRanks_ThrowsArgumentNullException()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        Assert.Throws<ArgumentNullException>(() =>
            new ProcessGroupLite(backend, null, myRank: 0, groupName: "test"));
    }

    [Fact]
    public void Constructor_MyRankNotInGroup_ThrowsArgumentException()
    {
        var backend = new MockCommunicationBackend(rank: 3, worldSize: 4);
        Assert.Throws<ArgumentException>(() =>
            new ProcessGroupLite(backend, new[] { 0, 1, 2 }, myRank: 3, groupName: "test"));
    }

    [Fact]
    public void GetGroupRank_ValidRank_ReturnsCorrectRank()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var group = new ProcessGroupLite(backend, new[] { 1, 3, 5 }, myRank: 1, groupName: "test");

        Assert.Equal(0, group.GetGroupRank(1));
        Assert.Equal(1, group.GetGroupRank(3));
        Assert.Equal(2, group.GetGroupRank(5));
    }

    [Fact]
    public void GetGroupRank_RankNotInGroup_ThrowsArgumentException()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var group = new ProcessGroupLite(backend, new[] { 1, 3, 5 }, myRank: 1, groupName: "test");

        Assert.Throws<ArgumentException>(() => group.GetGroupRank(2));
    }

    [Fact]
    public void CreateSubGroup_ValidRanks_CreatesSubGroup()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 8);
        var parentGroup = new ProcessGroupLite(backend, new[] { 0, 2, 4, 6 }, myRank: 0, groupName: "parent");
        var subGroup = parentGroup.CreateSubGroup("child", new[] { 0, 4 });

        Assert.Equal("child", subGroup.GroupName);
        Assert.Equal(0, subGroup.Rank);
        Assert.Equal(2, subGroup.GroupSize);
        Assert.True(subGroup.ContainsRank(0));
        Assert.True(subGroup.ContainsRank(4));
    }

    [Fact]
    public void CreateSubGroup_RankNotInParent_ThrowsArgumentException()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 8);
        var parentGroup = new ProcessGroupLite(backend, new[] { 0, 2, 4, 6 }, myRank: 0, groupName: "parent");

        Assert.Throws<ArgumentException>(() =>
            parentGroup.CreateSubGroup("child", new[] { 0, 1 }));
    }

    [Fact]
    public void IsWorldGroup_ReturnsCorrectValue()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);

        var worldGroup = new ProcessGroupLite(backend, new[] { 0, 1, 2, 3 }, myRank: 0, groupName: "world");
        Assert.True(worldGroup.IsWorldGroup);

        var customGroup = new ProcessGroupLite(backend, new[] { 0, 1 }, myRank: 0, groupName: "custom");
        Assert.False(customGroup.IsWorldGroup);
    }

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var group = new ProcessGroupLite(backend, new[] { 0, 1, 2 }, myRank: 0, groupName: "test");

        group.Dispose();
        group.Dispose(); // Should not throw
    }
}

public class ProcessGroupManagerTests
{
    [Fact]
    public void Constructor_ValidBackend_CreatesManager()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var manager = new ProcessGroupManager(backend);

        Assert.NotNull(manager.WorldGroup);
        Assert.Equal("world", manager.WorldGroup.GroupName);
        Assert.Equal(4, manager.WorldGroup.GroupSize);
        Assert.True(manager.WorldGroup.ContainsRank(0));
        Assert.True(manager.WorldGroup.ContainsRank(1));
        Assert.True(manager.WorldGroup.ContainsRank(2));
        Assert.True(manager.WorldGroup.ContainsRank(3));
    }

    [Fact]
    public void Constructor_NullBackend_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => new ProcessGroupManager(null));
    }

    [Fact]
    public void GetGroup_ExistingGroup_ReturnsGroup()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var manager = new ProcessGroupManager(backend);

        var group = manager.GetGroup("world");
        Assert.NotNull(group);
        Assert.Equal("world", group.GroupName);
    }

    [Fact]
    public void GetGroup_NonExistentGroup_ThrowsArgumentException()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var manager = new ProcessGroupManager(backend);

        Assert.Throws<ArgumentException>(() => manager.GetGroup("nonexistent"));
    }

    [Fact]
    public void GetGroup_NullGroupName_ThrowsArgumentNullException()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var manager = new ProcessGroupManager(backend);

        Assert.Throws<ArgumentNullException>(() => manager.GetGroup(null));
    }

    [Fact]
    public void HasGroup_ExistingGroup_ReturnsTrue()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var manager = new ProcessGroupManager(backend);

        Assert.True(manager.HasGroup("world"));
        Assert.False(manager.HasGroup("nonexistent"));
    }

    [Fact]
    public void CreateGroup_ValidParameters_CreatesGroup()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var manager = new ProcessGroupManager(backend);

        var group = manager.CreateGroup("test", new[] { 0, 1 });

        Assert.NotNull(group);
        Assert.Equal("test", group.GroupName);
        Assert.Equal(2, group.GroupSize);
        Assert.True(manager.HasGroup("test"));
    }

    [Fact]
    public void CreateGroup_DuplicateGroupName_ThrowsArgumentException()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var manager = new ProcessGroupManager(backend);

        manager.CreateGroup("test", new[] { 0, 1 });

        Assert.Throws<ArgumentException>(() => manager.CreateGroup("test", new[] { 2, 3 }));
    }

    [Fact]
    public void CreateGroup_EmptyRankList_ThrowsArgumentException()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var manager = new ProcessGroupManager(backend);

        Assert.Throws<ArgumentException>(() => manager.CreateGroup("test", Array.Empty<int>()));
    }

    [Fact]
    public void CreateGroup_NullRankList_ThrowsArgumentNullException()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var manager = new ProcessGroupManager(backend);

        Assert.Throws<ArgumentNullException>(() => manager.CreateGroup("test", null));
    }

    [Fact]
    public void CreateGroup_WithRange_CreatesGroup()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 8);
        var manager = new ProcessGroupManager(backend);

        var group = manager.CreateGroup("range", 2, 5);

        Assert.NotNull(group);
        Assert.Equal(4, group.GroupSize);
        Assert.True(group.ContainsRank(2));
        Assert.True(group.ContainsRank(3));
        Assert.True(group.ContainsRank(4));
        Assert.True(group.ContainsRank(5));
    }

    [Fact]
    public void CreateGroup_InvalidRange_ThrowsArgumentException()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 8);
        var manager = new ProcessGroupManager(backend);

        Assert.Throws<ArgumentException>(() => manager.CreateGroup("test", -1, 5));
        Assert.Throws<ArgumentException>(() => manager.CreateGroup("test", 2, 8));
        Assert.Throws<ArgumentException>(() => manager.CreateGroup("test", 5, 2));
    }

    [Fact]
    public void DestroyGroup_ValidGroup_RemovesGroup()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var manager = new ProcessGroupManager(backend);

        manager.CreateGroup("test", new[] { 0, 1 });
        Assert.True(manager.HasGroup("test"));

        manager.DestroyGroup("test");
        Assert.False(manager.HasGroup("test"));
    }

    [Fact]
    public void DestroyGroup_NonExistentGroup_ThrowsArgumentException()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var manager = new ProcessGroupManager(backend);

        Assert.Throws<ArgumentException>(() => manager.DestroyGroup("nonexistent"));
    }

    [Fact]
    public void DestroyGroup_WorldGroup_ThrowsArgumentException()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var manager = new ProcessGroupManager(backend);

        Assert.Throws<ArgumentException>(() => manager.DestroyGroup("world"));
    }

    [Fact]
    public void ListGroups_ReturnsAllGroupNames()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var manager = new ProcessGroupManager(backend);

        manager.CreateGroup("test1", new[] { 0 });
        manager.CreateGroup("test2", new[] { 1 });

        var groups = manager.ListGroups();

        Assert.Contains("world", groups);
        Assert.Contains("test1", groups);
        Assert.Contains("test2", groups);
    }

    [Fact]
    public void Dispose_DisposesAllNonWorldGroups()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var manager = new ProcessGroupManager(backend);

        manager.CreateGroup("test1", new[] { 0, 1 });
        manager.CreateGroup("test2", new[] { 2, 3 });

        manager.Dispose();

        Assert.True(manager.HasGroup("world"));
        Assert.False(manager.HasGroup("test1"));
        Assert.False(manager.HasGroup("test2"));
    }

    [Fact]
    public void Dispose_CanBeCalledMultipleTimes()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 4);
        var manager = new ProcessGroupManager(backend);

        manager.Dispose();
        manager.Dispose(); // Should not throw
    }
}

public class ProcessGroupHelpersTests
{
    [Fact]
    public void CreateSplitGroups_EvenSplit_CreatesGroups()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 8);
        var manager = new ProcessGroupManager(backend);

        var groups = ProcessGroupHelpers.CreateSplitGroups(manager, numGroups: 4, prefix: "split");

        Assert.Equal(4, groups.Count);
        Assert.Equal(2, groups[0].GroupSize);
        Assert.Equal(2, groups[1].GroupSize);
        Assert.Equal(2, groups[2].GroupSize);
        Assert.Equal(2, groups[3].GroupSize);

        Assert.True(groups[0].ContainsRank(0));
        Assert.True(groups[0].ContainsRank(1));
        Assert.True(groups[1].ContainsRank(2));
        Assert.True(groups[1].ContainsRank(3));
        Assert.True(groups[2].ContainsRank(4));
        Assert.True(groups[2].ContainsRank(5));
        Assert.True(groups[3].ContainsRank(6));
        Assert.True(groups[3].ContainsRank(7));
    }

    [Fact]
    public void CreateSplitGroups_UnevenSplit_DistributesRemainder()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 10);
        var manager = new ProcessGroupManager(backend);

        var groups = ProcessGroupHelpers.CreateSplitGroups(manager, numGroups: 3, prefix: "split");

        Assert.Equal(3, groups.Count);
        Assert.Equal(4, groups[0].GroupSize); // First group gets remainder
        Assert.Equal(3, groups[1].GroupSize);
        Assert.Equal(3, groups[2].GroupSize);

        Assert.True(groups[0].ContainsRank(0));
        Assert.True(groups[0].ContainsRank(1));
        Assert.True(groups[0].ContainsRank(2));
        Assert.True(groups[0].ContainsRank(3));
    }

    [Fact]
    public void CreatePipelineGroups_CreatesPipelineGroups()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 8);
        var manager = new ProcessGroupManager(backend);

        var groups = ProcessGroupHelpers.CreatePipelineGroups(manager, numPipelineStages: 4, prefix: "pipeline");

        Assert.Equal(4, groups.Count);
        Assert.Equal(2, groups[0].GroupSize);
        Assert.Equal(2, groups[1].GroupSize);
        Assert.Equal(2, groups[2].GroupSize);
        Assert.Equal(2, groups[3].GroupSize);

        Assert.True(groups[0].ContainsRank(0));
        Assert.True(groups[0].ContainsRank(4));
        Assert.True(groups[1].ContainsRank(1));
        Assert.True(groups[1].ContainsRank(5));
        Assert.True(groups[2].ContainsRank(2));
        Assert.True(groups[2].ContainsRank(6));
        Assert.True(groups[3].ContainsRank(3));
        Assert.True(groups[3].ContainsRank(7));
    }

    [Fact]
    public void CreatePipelineGroups_OddWorldSize_DistributesCorrectly()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 10);
        var manager = new ProcessGroupManager(backend);

        var groups = ProcessGroupHelpers.CreatePipelineGroups(manager, numPipelineStages: 3, prefix: "pipeline");

        Assert.Equal(3, groups.Count);
        Assert.Equal(4, groups[0].GroupSize); // 0, 3, 6, 9
        Assert.Equal(3, groups[1].GroupSize); // 1, 4, 7
        Assert.Equal(3, groups[2].GroupSize); // 2, 5, 8

        Assert.True(groups[0].ContainsRank(0));
        Assert.True(groups[0].ContainsRank(3));
        Assert.True(groups[0].ContainsRank(6));
        Assert.True(groups[0].ContainsRank(9));
    }

    [Fact]
    public void CreateDataParallelGroups_CreatesNodeGroups()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 8);
        var manager = new ProcessGroupManager(backend);

        var groups = ProcessGroupHelpers.CreateDataParallelGroups(manager, gpusPerNode: 4, prefix: "dp");

        Assert.Equal(2, groups.Count);
        Assert.Equal(4, groups[0].GroupSize);
        Assert.Equal(4, groups[1].GroupSize);

        // First node: ranks 0-3
        Assert.True(groups[0].ContainsRank(0));
        Assert.True(groups[0].ContainsRank(1));
        Assert.True(groups[0].ContainsRank(2));
        Assert.True(groups[0].ContainsRank(3));

        // Second node: ranks 4-7
        Assert.True(groups[1].ContainsRank(4));
        Assert.True(groups[1].ContainsRank(5));
        Assert.True(groups[1].ContainsRank(6));
        Assert.True(groups[1].ContainsRank(7));
    }

    [Fact]
    public void CreateDataParallelGroups_DifferentGpusPerNode_CreatesCorrectGroups()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 12);
        var manager = new ProcessGroupManager(backend);

        var groups = ProcessGroupHelpers.CreateDataParallelGroups(manager, gpusPerNode: 3, prefix: "dp");

        Assert.Equal(4, groups.Count);
        Assert.All(groups.Values, group => Assert.Equal(3, group.GroupSize));

        // Verify each node has correct ranks
        for (int node = 0; node < 4; node++)
        {
            for (int gpu = 0; gpu < 3; gpu++)
            {
                Assert.True(groups[node].ContainsRank(node * 3 + gpu));
            }
        }
    }

    [Fact]
    public void CreateDataParallelGroups_AllHelpersCreateCorrectGroupNames()
    {
        var backend = new MockCommunicationBackend(rank: 0, worldSize: 8);
        var manager = new ProcessGroupManager(backend);

        var splitGroups = ProcessGroupHelpers.CreateSplitGroups(manager, numGroups: 2, prefix: "split");
        var pipelineGroups = ProcessGroupHelpers.CreatePipelineGroups(manager, numPipelineStages: 2, prefix: "pipeline");
        var dpGroups = ProcessGroupHelpers.CreateDataParallelGroups(manager, gpusPerNode: 4, prefix: "dp");

        Assert.True(manager.HasGroup("split_0"));
        Assert.True(manager.HasGroup("split_1"));
        Assert.True(manager.HasGroup("pipeline_0"));
        Assert.True(manager.HasGroup("pipeline_1"));
        Assert.True(manager.HasGroup("dp_0"));
        Assert.True(manager.HasGroup("dp_1"));
    }
}
