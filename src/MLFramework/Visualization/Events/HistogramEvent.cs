using System.Text.Json;

namespace MachineLearning.Visualization.Events;

/// <summary>
/// Event representing histogram data (e.g., weight/gradient distributions)
/// </summary>
public class HistogramEvent : Event
{
    /// <summary>
    /// Name of the histogram
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Array of histogram values
    /// </summary>
    public float[] Values { get; }

    /// <summary>
    /// Training step at which this histogram was recorded
    /// </summary>
    public long Step { get; }

    /// <summary>
    /// Number of bins in the histogram
    /// </summary>
    public int BinCount { get; }

    /// <summary>
    /// Whether the histogram uses logarithmic bins
    /// </summary>
    public bool UseLogScale { get; }

    /// <summary>
    /// Minimum value for binning
    /// </summary>
    public float Min { get; }

    /// <summary>
    /// Maximum value for binning
    /// </summary>
    public float Max { get; }

    /// <summary>
    /// Additional metadata for this histogram
    /// </summary>
    public Dictionary<string, string> Metadata { get; }

    /// <summary>
    /// Creates a new histogram event
    /// </summary>
    /// <param name="name">Name of the histogram</param>
    /// <param name="values">Array of histogram values</param>
    /// <param name="step">Training step</param>
    /// <param name="binCount">Number of bins</param>
    /// <param name="useLogScale">Whether to use log scale</param>
    /// <param name="min">Minimum value for binning</param>
    /// <param name="max">Maximum value for binning</param>
    /// <param name="metadata">Optional metadata</param>
    public HistogramEvent(
        string name,
        float[] values,
        long step = -1,
        int binCount = 30,
        bool useLogScale = false,
        float min = float.MinValue,
        float max = float.MaxValue,
        Dictionary<string, string>? metadata = null)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Values = values ?? throw new ArgumentNullException(nameof(values));
        Step = step;
        BinCount = binCount;
        UseLogScale = useLogScale;
        Min = min;
        Max = max;
        Metadata = metadata ?? new Dictionary<string, string>();
    }

    /// <summary>
    /// Serializes the event to bytes
    /// </summary>
    public override byte[] Serialize()
    {
        var data = new Dictionary<string, object>
        {
            { "type", MachineLearning.Visualization.Events.EventType.Histogram.ToString() },
            { "name", Name },
            { "values", Values },
            { "step", Step },
            { "binCount", BinCount },
            { "useLogScale", UseLogScale },
            { "min", Min },
            { "max", Max },
            { "timestamp", Timestamp.ToString("o") },
            { "eventId", EventId.ToString() },
            { "metadata", Metadata }
        };

        return JsonSerializer.SerializeToUtf8Bytes(data);
    }

    /// <summary>
    /// Deserializes bytes back to an event
    /// </summary>
    public override void Deserialize(byte[] data)
    {
        // Note: In a real implementation, we would update the properties from the data
        // This is a placeholder that shows the concept
        var json = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(data);
        // Implementation would parse and update properties
    }
}
