using System.Text;
using MachineLearning.Visualization.Events;
using MLFramework.Visualization.Histograms.Statistics;

namespace MachineLearning.Visualization.Histograms;

/// <summary>
/// Event type for histogram data
/// </summary>
public enum HistogramEventType
{
    /// <summary>
    /// Basic histogram data
    /// </summary>
    Histogram = 0,

    /// <summary>
    /// Tensor distribution with extended statistics
    /// </summary>
    Distribution = 1
}

/// <summary>
/// Event for histogram and tensor distribution data
/// </summary>
public class HistogramEvent : Event
{
    /// <summary>
    /// Gets the type of histogram event
    /// </summary>
    public HistogramEventType EventType { get; }

    /// <summary>
    /// Gets the name of the histogram/distribution
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the training step
    /// </summary>
    public long Step { get; }

    /// <summary>
    /// Gets the bin edges (for histogram events)
    /// </summary>
    public float[] BinEdges { get; }

    /// <summary>
    /// Gets the bin counts (for histogram events)
    /// </summary>
    public long[] BinCounts { get; }

    /// <summary>
    /// Gets the minimum value
    /// </summary>
    public float Min { get; }

    /// <summary>
    /// Gets the maximum value
    /// </summary>
    public float Max { get; }

    /// <summary>
    /// Gets the mean value
    /// </summary>
    public float Mean { get; }

    /// <summary>
    /// Gets the standard deviation
    /// </summary>
    public float Std { get; }

    /// <summary>
    /// Gets the quantiles (for distribution events)
    /// </summary>
    public float[] Quantiles { get; }

    /// <summary>
    /// Gets the median (for distribution events)
    /// </summary>
    public float Median { get; }

    /// <summary>
    /// Gets the skewness (for distribution events)
    /// </summary>
    public float Skewness { get; }

    /// <summary>
    /// Gets the kurtosis (for distribution events)
    /// </summary>
    public float Kurtosis { get; }

    /// <summary>
    /// Gets the dead neuron count (for distribution events)
    /// </summary>
    public int DeadNeuronCount { get; }

    /// <summary>
    /// Gets the outlier count (for distribution events)
    /// </summary>
    public int OutlierCount { get; }

    /// <summary>
    /// Private constructor for deserialization
    /// </summary>
    private HistogramEvent()
    {
    }

    /// <summary>
    /// Creates a histogram event from HistogramData
    /// </summary>
    public HistogramEvent(HistogramData histogramData)
    {
        if (histogramData == null)
        {
            throw new ArgumentNullException(nameof(histogramData));
        }

        EventType = HistogramEventType.Histogram;
        Name = histogramData.Name;
        Step = histogramData.Step;
        BinEdges = histogramData.BinEdges;
        BinCounts = histogramData.BinCounts;
        Min = histogramData.Min;
        Max = histogramData.Max;
        Mean = histogramData.Mean;
        Std = histogramData.Std;
        Quantiles = histogramData.Quantiles;
        Median = histogramData.Quantiles[2]; // 50th percentile
        Skewness = 0;
        Kurtosis = 0;
        DeadNeuronCount = 0;
        OutlierCount = 0;
    }

    /// <summary>
    /// Creates a histogram event from TensorDistribution
    /// </summary>
    public HistogramEvent(TensorDistribution distribution)
    {
        if (distribution == null)
        {
            throw new ArgumentNullException(nameof(distribution));
        }

        EventType = HistogramEventType.Distribution;
        Name = distribution.Name;
        Step = distribution.Step;
        BinEdges = distribution.Histogram.BinEdges;
        BinCounts = distribution.Histogram.BinCounts;
        Min = distribution.Min;
        Max = distribution.Max;
        Mean = distribution.Mean;
        Std = distribution.Std;
        Quantiles = distribution.Quantiles;
        Median = distribution.Median;
        Skewness = distribution.Skewness;
        Kurtosis = distribution.Kurtosis;
        DeadNeuronCount = distribution.DeadNeuronCount;
        OutlierCount = distribution.OutlierCount;
    }

    /// <summary>
    /// Serializes the event to bytes for storage
    /// </summary>
    public override byte[] Serialize()
    {
        var sb = new StringBuilder();

        // Header
        sb.AppendLine($"{Timestamp:o}");
        sb.AppendLine($"{EventId}");
        sb.AppendLine($"{(int)EventType}");
        sb.AppendLine($"{Name}");

        // Step
        sb.AppendLine($"{Step}");

        // Basic statistics
        sb.AppendLine($"{Min}");
        sb.AppendLine($"{Max}");
        sb.AppendLine($"{Mean}");
        sb.AppendLine($"{Std}");
        sb.AppendLine($"{Median}");

        // Bin data
        sb.AppendLine($"{BinEdges.Length}");
        for (int i = 0; i < BinEdges.Length; i++)
        {
            sb.AppendLine($"{BinEdges[i]}");
        }

        sb.AppendLine($"{BinCounts.Length}");
        for (int i = 0; i < BinCounts.Length; i++)
        {
            sb.AppendLine($"{BinCounts[i]}");
        }

        // Extended statistics (for distribution events)
        sb.AppendLine($"{Skewness}");
        sb.AppendLine($"{Kurtosis}");
        sb.AppendLine($"{DeadNeuronCount}");
        sb.AppendLine($"{OutlierCount}");

        // Quantiles
        sb.AppendLine($"{Quantiles.Length}");
        for (int i = 0; i < Quantiles.Length; i++)
        {
            sb.AppendLine($"{Quantiles[i]}");
        }

        return Encoding.UTF8.GetBytes(sb.ToString());
    }

    /// <summary>
    /// Deserializes bytes back to an event
    /// </summary>
    public override void Deserialize(byte[] data)
    {
        if (data == null || data.Length == 0)
        {
            throw new ArgumentException("Data cannot be null or empty", nameof(data));
        }

        var lines = Encoding.UTF8.GetString(data).Split(new[] { Environment.NewLine }, StringSplitOptions.RemoveEmptyEntries);
        if (lines.Length < 15)
        {
            throw new ArgumentException("Invalid data format", nameof(data));
        }

        var lineIndex = 0;
        Timestamp = DateTime.ParseExact(lines[lineIndex++], "o", null);
        var eventId = Guid.Parse(lines[lineIndex++]);
        EventType = (HistogramEventType)int.Parse(lines[lineIndex++]);
        Name = lines[lineIndex++];
        Step = long.Parse(lines[lineIndex++]);
        Min = float.Parse(lines[lineIndex++]);
        Max = float.Parse(lines[lineIndex++]);
        Mean = float.Parse(lines[lineIndex++]);
        Std = float.Parse(lines[lineIndex++]);
        Median = float.Parse(lines[lineIndex++]);

        // Bin edges
        int binEdgesCount = int.Parse(lines[lineIndex++]);
        BinEdges = new float[binEdgesCount];
        for (int i = 0; i < binEdgesCount; i++)
        {
            BinEdges[i] = float.Parse(lines[lineIndex++]);
        }

        // Bin counts
        int binCountsCount = int.Parse(lines[lineIndex++]);
        BinCounts = new long[binCountsCount];
        for (int i = 0; i < binCountsCount; i++)
        {
            BinCounts[i] = long.Parse(lines[lineIndex++]);
        }

        // Extended statistics
        Skewness = float.Parse(lines[lineIndex++]);
        Kurtosis = float.Parse(lines[lineIndex++]);
        DeadNeuronCount = int.Parse(lines[lineIndex++]);
        OutlierCount = int.Parse(lines[lineIndex++]);

        // Quantiles
        int quantilesCount = int.Parse(lines[lineIndex++]);
        Quantiles = new float[quantilesCount];
        for (int i = 0; i < quantilesCount; i++)
        {
            Quantiles[i] = float.Parse(lines[lineIndex++]);
        }

        // Set the EventId using reflection since it's read-only
        var eventIdProperty = typeof(Event).GetProperty(nameof(EventId));
        eventIdProperty?.SetValue(this, eventId);
    }
}
