namespace MLFramework.Inference.ContinuousBatching.Prefill;

/// <summary>
/// Cache entry for prefill results.
/// </summary>
public class PrefillCacheEntry
{
    /// <summary>
    /// The cached prompt.
    /// </summary>
    public string Prompt { get; }

    /// <summary>
    /// Tokenized prompt.
    /// </summary>
    public int[] Tokens { get; }

    /// <summary>
    /// Logits from the prefill pass.
    /// </summary>
    public float[] Logits { get; }

    /// <summary>
    /// Time when the entry was created.
    /// </summary>
    public DateTime CreatedTime { get; }

    /// <summary>
    /// Time when the entry was last accessed.
    /// </summary>
    public DateTime LastAccessedTime { get; private set; }

    /// <summary>
    /// Number of times this entry has been accessed.
    /// </summary>
    public int AccessCount { get; private set; }

    /// <summary>
    /// Creates a new prefill cache entry.
    /// </summary>
    public PrefillCacheEntry(string prompt, int[] tokens, float[] logits)
    {
        Prompt = prompt;
        Tokens = tokens;
        Logits = logits;
        CreatedTime = DateTime.UtcNow;
        LastAccessedTime = DateTime.UtcNow;
        AccessCount = 0;
    }

    /// <summary>
    /// Marks this entry as accessed.
    /// </summary>
    public void MarkAccessed()
    {
        LastAccessedTime = DateTime.UtcNow;
        AccessCount++;
    }

    /// <summary>
    /// Checks if this entry has expired based on the TTL.
    /// </summary>
    /// <param name="ttl">The time-to-live duration.</param>
    /// <returns>True if the entry is expired, false otherwise.</returns>
    public bool IsExpired(TimeSpan ttl) =>
        DateTime.UtcNow - LastAccessedTime > ttl;
}
