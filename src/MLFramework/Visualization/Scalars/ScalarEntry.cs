namespace MachineLearning.Visualization.Scalars;

/// <summary>
/// Represents a single scalar metric entry with step and timestamp information
/// </summary>
public class ScalarEntry
{
    /// <summary>
    /// Training step at which this entry was recorded
    /// </summary>
    public long Step { get; }

    /// <summary>
    /// Value of the metric
    /// </summary>
    public float Value { get; }

    /// <summary>
    /// Timestamp when the entry was recorded
    /// </summary>
    public DateTime Timestamp { get; }

    /// <summary>
    /// Creates a new scalar entry
    /// </summary>
    /// <param name="step">Training step</param>
    /// <param name="value">Value of the metric</param>
    /// <param name="timestamp">Timestamp (defaults to current time)</param>
    public ScalarEntry(long step, float value, DateTime? timestamp = null)
    {
        Step = step;
        Value = value;
        Timestamp = timestamp ?? DateTime.UtcNow;
    }
}
