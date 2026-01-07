namespace MLFramework.Inference.ContinuousBatching;

/// <summary>
/// Counter that maintains a sliding window of events.
/// </summary>
public class SlidingWindowCounter
{
    private readonly Queue<(DateTime Timestamp, int Count)> _events;
    private readonly TimeSpan _window;

    /// <summary>
    /// Creates a new sliding window counter.
    /// </summary>
    /// <param name="windowSeconds">Window duration in seconds.</param>
    public SlidingWindowCounter(int windowSeconds)
    {
        _events = new Queue<(DateTime, int)>();
        _window = TimeSpan.FromSeconds(windowSeconds);
    }

    /// <summary>
    /// Adds a count event to the window.
    /// </summary>
    /// <param name="count">Count to add.</param>
    public void Add(int count)
    {
        _events.Enqueue((DateTime.UtcNow, count));
        CleanOldEvents();
    }

    /// <summary>
    /// Gets the rate (count per second) over the sliding window.
    /// </summary>
    /// <returns>Rate per second.</returns>
    public double GetRate()
    {
        CleanOldEvents();
        int total = _events.Sum(e => e.Count);
        return total / _window.TotalSeconds;
    }

    /// <summary>
    /// Resets the counter, clearing all events.
    /// </summary>
    public void Reset()
    {
        _events.Clear();
    }

    /// <summary>
    /// Removes events that are outside the sliding window.
    /// </summary>
    private void CleanOldEvents()
    {
        var cutoff = DateTime.UtcNow - _window;
        while (_events.Count > 0 && _events.Peek().Timestamp < cutoff)
        {
            _events.Dequeue();
        }
    }
}
