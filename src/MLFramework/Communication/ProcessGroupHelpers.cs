namespace MLFramework.Communication;

using System.Collections.Generic;
using System.Linq;

/// <summary>
/// Helper methods for creating common process group patterns
/// </summary>
public static class ProcessGroupHelpers
{
    /// <summary>
    /// Split world into N equal-sized groups
    /// </summary>
    /// <returns>Dictionary mapping group index to process group</returns>
    public static Dictionary<int, ProcessGroupLite> CreateSplitGroups(
        ProcessGroupManager manager,
        int numGroups,
        string prefix = "split")
    {
        var world = manager.WorldGroup;
        var worldSize = world.GroupSize;
        var groupsPerGroup = worldSize / numGroups;
        var remainder = worldSize % numGroups;

        var groups = new Dictionary<int, ProcessGroupLite>();

        int startRank = 0;
        for (int i = 0; i < numGroups; i++)
        {
            int groupSize = groupsPerGroup + (i < remainder ? 1 : 0);
            int endRank = startRank + groupSize - 1;

            string groupName = $"{prefix}_{i}";
            var group = manager.CreateGroup(groupName, startRank, endRank);
            groups[i] = group;

            startRank += groupSize;
        }

        return groups;
    }

    /// <summary>
    /// Create pipeline groups (alternating ranks)
    /// </summary>
    public static Dictionary<int, ProcessGroupLite> CreatePipelineGroups(
        ProcessGroupManager manager,
        int numPipelineStages,
        string prefix = "pipeline")
    {
        var world = manager.WorldGroup;
        var worldSize = world.GroupSize;
        var groups = new Dictionary<int, ProcessGroupLite>();

        for (int stage = 0; stage < numPipelineStages; stage++)
        {
            var stageRanks = Enumerable.Range(stage, (worldSize - stage + numPipelineStages - 1) / numPipelineStages)
                                     .Where(r => r < worldSize)
                                     .Select(r => r * numPipelineStages + stage)
                                     .Where(r => r < worldSize);

            string groupName = $"{prefix}_{stage}";
            var group = manager.CreateGroup(groupName, stageRanks);
            groups[stage] = group;
        }

        return groups;
    }

    /// <summary>
    /// Create groups for data parallelism (each GPU in a node)
    /// </summary>
    /// <param name="gpusPerNode">Number of GPUs per compute node</param>
    public static Dictionary<int, ProcessGroupLite> CreateDataParallelGroups(
        ProcessGroupManager manager,
        int gpusPerNode,
        string prefix = "dp")
    {
        var world = manager.WorldGroup;
        var worldSize = world.GroupSize;
        var numNodes = worldSize / gpusPerNode;

        var groups = new Dictionary<int, ProcessGroupLite>();

        for (int node = 0; node < numNodes; node++)
        {
            var nodeRanks = Enumerable.Range(node * gpusPerNode, gpusPerNode);
            string groupName = $"{prefix}_{node}";
            var group = manager.CreateGroup(groupName, nodeRanks);
            groups[node] = group;
        }

        return groups;
    }
}
