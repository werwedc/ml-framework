namespace MLFramework.Communication;

using System;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// Represents a group of ranks for collective operations (lightweight version)
/// </summary>
public class ProcessGroupLite : IDisposable
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
    public IReadOnlyCollection<int> Ranks { get { return _ranks.ToList().AsReadOnly(); } }

    /// <summary>
    /// True if this is default world group
    /// </summary>
    public bool IsWorldGroup { get { return _groupName == "world"; } }

    /// <summary>
    /// Creates a new process group
    /// </summary>
    /// <param name="backend">Communication backend</param>
    /// <param name="ranks">Ranks in this group</param>
    /// <param name="myRank">This process's rank</param>
    /// <param name="groupName">Unique group identifier</param>
    public ProcessGroupLite(
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
    public ProcessGroupLite CreateSubGroup(string name, IEnumerable<int> worldRanks)
    {
        // Validate all ranks are in this group
        foreach (var rank in worldRanks)
        {
            if (!_ranks.Contains(rank))
            {
                throw new ArgumentException($"Rank {rank} is not in group {_groupName}");
            }
        }

        return new ProcessGroupLite(_backend, worldRanks, _backend.Rank, name);
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
