# Spec: Process Group Management

## Overview
Implement process group management to organize ranks into logical groups for fine-grained control over communication.

## Dependencies
- `spec_communication_interfaces.md`

## Technical Requirements

### 1. ProcessGroup Class
Create a class to represent a group of ranks that can communicate with each other.

```csharp
namespace MLFramework.Communication
{
    /// <summary>
    /// Represents a group of ranks for collective operations
    /// </summary>
    public class ProcessGroup : IDisposable
    {
        private readonly ISet<int> _ranks;
        private readonly int _groupRank;
        private readonly ICommunicationBackend _backend;
        private readonly string _groupName;
        private bool _disposed;

        /// <summary>
        /// Unique identifier for this process group
        /// </summary>
        public string GroupName { get { return _groupName; } }

        /// <summary>
        /// Rank of this process within the group
        /// </summary>
        public int Rank { get { return _groupRank; } }

        /// <summary>
        /// Number of processes in this group
        /// </summary>
        public int GroupSize { get { return _ranks.Count; } }

        /// <summary>
        /// All ranks in this group
        /// </summary>
        public IReadOnlyCollection<int> Ranks { get { return _ranks; } }

        /// <summary>
        /// True if this is the default world group
        /// </summary>
        public bool IsWorldGroup { get { return _groupName == "world"; } }

        /// <summary>
        /// Creates a new process group
        /// </summary>
        /// <param name="backend">Communication backend</param>
        /// <param name="ranks">Ranks in this group</param>
        /// <param name="myRank">This process's rank</param>
        /// <param name="groupName">Unique group identifier</param>
        public ProcessGroup(
            ICommunicationBackend backend,
            IEnumerable<int> ranks,
            int myRank,
            string groupName)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
            _groupName = groupName ?? throw new ArgumentNullException(nameof(groupName));

            _ranks = new HashSet<int>(ranks ?? throw new ArgumentNullException(nameof(ranks)));

            if (!_ranks.Contains(myRank))
            {
                throw new ArgumentException($"My rank {myRank} is not in the group");
            }

            _groupRank = _ranks.OrderBy(r => r).ToList().IndexOf(myRank);
        }

        /// <summary>
        /// Check if a specific rank is in this group
        /// </summary>
        public bool ContainsRank(int rank)
        {
            return _ranks.Contains(rank);
        }

        /// <summary>
        /// Get the group-local rank for a world rank
        /// </summary>
        public int GetGroupRank(int worldRank)
        {
            if (!_ranks.Contains(worldRank))
            {
                throw new ArgumentException($"Rank {worldRank} is not in group {_groupName}");
            }

            return _ranks.OrderBy(r => r).ToList().IndexOf(worldRank);
        }

        /// <summary>
        /// Create a subgroup from this group
        /// </summary>
        public ProcessGroup CreateSubGroup(string name, IEnumerable<int> worldRanks)
        {
            // Validate all ranks are in this group
            foreach (var rank in worldRanks)
            {
                if (!_ranks.Contains(rank))
                {
                    throw new ArgumentException($"Rank {rank} is not in group {_groupName}");
                }
            }

            return new ProcessGroup(_backend, worldRanks, _backend.Rank, name);
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                // Cleanup resources
                _disposed = true;
            }
        }
    }
}
```

### 2. ProcessGroupManager Class
Manage the lifecycle of process groups, including the default world group.

```csharp
namespace MLFramework.Communication
{
    /// <summary>
    /// Manages process groups for distributed communication
    /// </summary>
    public class ProcessGroupManager : IDisposable
    {
        private readonly ICommunicationBackend _backend;
        private readonly Dictionary<string, ProcessGroup> _groups;
        private readonly ProcessGroup _worldGroup;
        private bool _disposed;

        /// <summary>
        /// Gets the default world group (all ranks)
        /// </summary>
        public ProcessGroup WorldGroup { get { return _worldGroup; } }

        /// <summary>
        /// Creates a new process group manager
        /// </summary>
        public ProcessGroupManager(ICommunicationBackend backend)
        {
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
            _groups = new Dictionary<string, ProcessGroup>();

            // Create the default world group containing all ranks
            var allRanks = Enumerable.Range(0, backend.WorldSize);
            _worldGroup = new ProcessGroup(backend, allRanks, backend.Rank, "world");
            _groups["world"] = _worldGroup;
        }

        /// <summary>
        /// Get or create a process group by name
        /// </summary>
        public ProcessGroup GetGroup(string groupName)
        {
            if (groupName == null)
            {
                throw new ArgumentNullException(nameof(groupName));
            }

            if (!_groups.TryGetValue(groupName, out var group))
            {
                throw new ArgumentException($"Process group '{groupName}' does not exist");
            }

            return group;
        }

        /// <summary>
        /// Check if a process group exists
        /// </summary>
        public bool HasGroup(string groupName)
        {
            return _groups.ContainsKey(groupName);
        }

        /// <summary>
        /// Create a new process group
        /// </summary>
        /// <param name="groupName">Unique group name</param>
        /// <param name="ranks">Ranks to include in the group</param>
        /// <param name="globalRanks">True if ranks are global world ranks, false if they're relative to world</param>
        public ProcessGroup CreateGroup(string groupName, IEnumerable<int> ranks, bool globalRanks = true)
        {
            if (groupName == null)
            {
                throw new ArgumentNullException(nameof(groupName));
            }

            if (_groups.ContainsKey(groupName))
            {
                throw new ArgumentException($"Process group '{groupName}' already exists");
            }

            if (ranks == null)
            {
                throw new ArgumentNullException(nameof(ranks));
            }

            var rankList = ranks.ToList();

            if (rankList.Count == 0)
            {
                throw new ArgumentException("Process group must contain at least one rank");
            }

            // Use global ranks or convert from world-relative ranks
            var worldRanks = globalRanks ? rankList : rankList.Select(r => _worldGroup.GetGroupRank(r));

            var group = new ProcessGroup(_backend, worldRanks, _backend.Rank, groupName);
            _groups[groupName] = group;

            return group;
        }

        /// <summary>
        /// Create a process group by rank range
        /// </summary>
        public ProcessGroup CreateGroup(string groupName, int startRank, int endRank)
        {
            if (startRank < 0 || endRank >= _backend.WorldSize || startRank > endRank)
            {
                throw new ArgumentException($"Invalid rank range [{startRank}, {endRank}]");
            }

            var ranks = Enumerable.Range(startRank, endRank - startRank + 1);
            return CreateGroup(groupName, ranks);
        }

        /// <summary>
        /// Destroy a process group
        /// </summary>
        public void DestroyGroup(string groupName)
        {
            if (groupName == "world")
            {
                throw new ArgumentException("Cannot destroy the world group");
            }

            if (!_groups.TryGetValue(groupName, out var group))
            {
                throw new ArgumentException($"Process group '{groupName}' does not exist");
            }

            group.Dispose();
            _groups.Remove(groupName);
        }

        /// <summary>
        /// Get all group names
        /// </summary>
        public IEnumerable<string> ListGroups()
        {
            return _groups.Keys.ToList();
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                // Dispose all groups except world
                foreach (var kvp in _groups.ToList())
                {
                    if (kvp.Key != "world")
                    {
                        kvp.Value.Dispose();
                        _groups.Remove(kvp.Key);
                    }
                }

                _disposed = true;
            }
        }
    }
}
```

### 3. Group Creation Helpers
Provide helper methods to create common group patterns.

```csharp
namespace MLFramework.Communication
{
    /// <summary>
    /// Helper methods for creating common process group patterns
    /// </summary>
    public static class ProcessGroupHelpers
    {
        /// <summary>
        /// Split world into N equal-sized groups
        /// </summary>
        /// <returns>Dictionary mapping group index to process group</returns>
        public static Dictionary<int, ProcessGroup> CreateSplitGroups(
            ProcessGroupManager manager,
            int numGroups,
            string prefix = "split")
        {
            var world = manager.WorldGroup;
            var worldSize = world.GroupSize;
            var groupsPerGroup = worldSize / numGroups;
            var remainder = worldSize % numGroups;

            var groups = new Dictionary<int, ProcessGroup>();

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
        public static Dictionary<int, ProcessGroup> CreatePipelineGroups(
            ProcessGroupManager manager,
            int numPipelineStages,
            string prefix = "pipeline")
        {
            var world = manager.WorldGroup;
            var worldSize = world.GroupSize;
            var groups = new Dictionary<int, ProcessGroup>();

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
        public static Dictionary<int, ProcessGroup> CreateDataParallelGroups(
            ProcessGroupManager manager,
            int gpusPerNode,
            string prefix = "dp")
        {
            var world = manager.WorldGroup;
            var worldSize = world.GroupSize;
            var numNodes = worldSize / gpusPerNode;

            var groups = new Dictionary<int, ProcessGroup>();

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
}
```

## Implementation Notes

1. **File Structure:**
   - `src/MLFramework/Communication/ProcessGroup.cs`
   - `src/MLFramework/Communication/ProcessGroupManager.cs`
   - `src/MLFramework/Communication/ProcessGroupHelpers.cs`

2. **Design Decisions:**
   - Process groups are lightweight wrappers around rank sets
   - The world group is always available and cannot be destroyed
   - Group names must be unique
   - Ranks are stored in sorted order for consistency

3. **Thread Safety:**
   - Consider adding locks if groups will be created/destroyed concurrently
   - The current implementation is not thread-safe

4. **Memory Management:**
   - Properly dispose of groups when no longer needed
   - ProcessGroupManager implements IDisposable for cleanup

## Testing Requirements
- Unit tests for ProcessGroup creation and validation
- Tests for ProcessGroupManager CRUD operations
- Tests for helper methods (split groups, pipeline groups)
- Edge case tests (empty groups, duplicate ranks, invalid ranges)

## Success Criteria
- Can create and manage process groups
- World group is always available
- Helper methods create valid group patterns
- Proper error handling for invalid inputs
