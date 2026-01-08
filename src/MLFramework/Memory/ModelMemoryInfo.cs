namespace MLFramework.Memory;

/// <summary>
/// Information about a loaded model's memory usage and access patterns.
/// </summary>
public class ModelMemoryInfo
{
    public string ModelName { get; }
    public string Version { get; }
    public long MemoryBytes { get; }
    public DateTime LastAccessTime { get; private set; }
    public int AccessCount { get; private set; }
    public bool IsPinned { get; set; }
    public long Weight { get; private set; }

    public ModelMemoryInfo(string modelName, string version, long memoryBytes)
    {
        ModelName = modelName;
        Version = version;
        MemoryBytes = memoryBytes;
        LastAccessTime = DateTime.UtcNow;
        AccessCount = 0;
        IsPinned = false;
        Weight = CalculateWeight(LastAccessTime, AccessCount);
    }

    /// <summary>
    /// Updates the last access time and increments access count.
    /// </summary>
    public void RecordAccess()
    {
        LastAccessTime = DateTime.UtcNow;
        AccessCount++;
        Weight = CalculateWeight(LastAccessTime, AccessCount);
    }

    /// <summary>
    /// Calculates the weight for eviction decisions (lower = more likely to evict).
    /// Combines LRU (last access time) and LFU (access count) heuristics.
    /// </summary>
    private static long CalculateWeight(DateTime lastAccessTime, int accessCount)
    {
        // Use ticks for time precision
        // Lower weight = less recently used + less frequently used
        var timeWeight = lastAccessTime.Ticks;
        var frequencyWeight = accessCount * TimeSpan.TicksPerDay; // Give frequency some weight but prioritize recency
        return timeWeight - frequencyWeight;
    }
}
