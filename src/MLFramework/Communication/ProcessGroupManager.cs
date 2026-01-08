namespace MLFramework.Communication;

using System;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// Manages process groups for distributed communication
/// </summary>
public class ProcessGroupManager : IDisposable
{
    private readonly ICommunicationBackend _backend;
    private readonly Dictionary<string, ProcessGroupLite> _groups;
    private readonly ProcessGroupLite _worldGroup;
    private bool _disposed;

    /// <summary>
    /// Gets the default world group (all ranks)
    /// </summary>
    public ProcessGroupLite WorldGroup { get { return _worldGroup; } }

    /// <summary>
    /// Creates a new process group manager
    /// </summary>
    public ProcessGroupManager(ICommunicationBackend backend)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _groups = new Dictionary<string, ProcessGroupLite>();

        // Create the default world group containing all ranks
        var allRanks = Enumerable.Range(0, backend.WorldSize);
        _worldGroup = new ProcessGroupLite(backend, allRanks, backend.Rank, "world");
        _groups["world"] = _worldGroup;
    }

    /// <summary>
    /// Get or create a process group by name
    /// </summary>
    public ProcessGroupLite GetGroup(string groupName)
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
    public ProcessGroupLite CreateGroup(string groupName, IEnumerable<int> ranks, bool globalRanks = true)
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

        var group = new ProcessGroupLite(_backend, worldRanks, _backend.Rank, groupName);
        _groups[groupName] = group;

        return group;
    }

    /// <summary>
    /// Create a process group by rank range
    /// </summary>
    public ProcessGroupLite CreateGroup(string groupName, int startRank, int endRank)
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
