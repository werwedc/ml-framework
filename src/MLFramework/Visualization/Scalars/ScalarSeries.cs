namespace MachineLearning.Visualization.Scalars;

/// <summary>
/// Represents a series of scalar metric entries for a single metric name
/// </summary>
public class ScalarSeries
{
    private readonly List<ScalarEntry> _entries;
    private readonly object _lock = new object();

    /// <summary>
    /// Name of the metric
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// All entries in the series
    /// </summary>
    public IReadOnlyList<ScalarEntry> Entries
    {
        get
        {
            lock (_lock)
            {
                return _entries.ToList().AsReadOnly();
            }
        }
    }

    /// <summary>
    /// Minimum value in the series
    /// </summary>
    public float? Min { get; private set; }

    /// <summary>
    /// Maximum value in the series
    /// </summary>
    public float? Max { get; private set; }

    /// <summary>
    /// Average value in the series
    /// </summary>
    public float Average { get; private set; }

    /// <summary>
    /// Number of entries in the series
    /// </summary>
    public int Count => _entries.Count;

    /// <summary>
    /// Creates a new scalar series
    /// </summary>
    /// <param name="name">Name of the metric</param>
    public ScalarSeries(string name)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        _entries = new List<ScalarEntry>();
    }

    /// <summary>
    /// Creates a new scalar series with initial entries
    /// </summary>
    /// <param name="name">Name of the metric</param>
    /// <param name="entries">Initial entries</param>
    public ScalarSeries(string name, IEnumerable<ScalarEntry> entries) : this(name)
    {
        if (entries == null) throw new ArgumentNullException(nameof(entries));
        _entries.AddRange(entries);
        UpdateStatistics();
    }

    /// <summary>
    /// Adds an entry to the series
    /// </summary>
    /// <param name="entry">Entry to add</param>
    public void Add(ScalarEntry entry)
    {
        if (entry == null) throw new ArgumentNullException(nameof(entry));

        lock (_lock)
        {
            _entries.Add(entry);
            UpdateStatisticsIncremental(entry.Value);
        }
    }

    /// <summary>
    /// Adds multiple entries to the series
    /// </summary>
    /// <param name="entries">Entries to add</param>
    public void AddRange(IEnumerable<ScalarEntry> entries)
    {
        if (entries == null) throw new ArgumentNullException(nameof(entries));

        lock (_lock)
        {
            foreach (var entry in entries)
            {
                _entries.Add(entry);
            }
            UpdateStatistics();
        }
    }

    /// <summary>
    /// Gets entries within a step range
    /// </summary>
    /// <param name="startStep">Starting step (inclusive)</param>
    /// <param name="endStep">Ending step (inclusive)</param>
    /// <returns>Entries within the range</returns>
    public IEnumerable<ScalarEntry> GetRange(long startStep, long endStep)
    {
        lock (_lock)
        {
            return _entries.Where(e => e.Step >= startStep && e.Step <= endStep).ToList();
        }
    }

    /// <summary>
    /// Returns a smoothed version of the series using a moving average
    /// </summary>
    /// <param name="windowSize">Size of the moving average window</param>
    /// <returns>New smoothed series</returns>
    public ScalarSeries Smoothed(int windowSize)
    {
        if (windowSize <= 0)
            throw new ArgumentException("Window size must be positive", nameof(windowSize));

        List<ScalarEntry> smoothedEntries;
        lock (_lock)
        {
            smoothedEntries = new List<ScalarEntry>(_entries.Count);

            for (int i = 0; i < _entries.Count; i++)
            {
                int start = Math.Max(0, i - windowSize / 2);
                int end = Math.Min(_entries.Count - 1, i + windowSize / 2);
                int count = end - start + 1;

                float sum = 0;
                for (int j = start; j <= end; j++)
                {
                    sum += _entries[j].Value;
                }

                float avg = sum / count;
                smoothedEntries.Add(new ScalarEntry(_entries[i].Step, avg, _entries[i].Timestamp));
            }
        }

        return new ScalarSeries(Name + "_smoothed", smoothedEntries);
    }

    /// <summary>
    /// Returns a resampled version of the series with a target number of points
    /// </summary>
    /// <param name="targetCount">Target number of points</param>
    /// <returns>New resampled series</returns>
    public ScalarSeries Resampled(int targetCount)
    {
        if (targetCount <= 0)
            throw new ArgumentException("Target count must be positive", nameof(targetCount));

        List<ScalarEntry> sourceEntries;
        lock (_lock)
        {
            sourceEntries = new List<ScalarEntry>(_entries);
        }

        if (sourceEntries.Count <= targetCount)
            return new ScalarSeries(Name + "_resampled", sourceEntries);

        var resampledEntries = new List<ScalarEntry>(targetCount);
        float stepSize = (float)(sourceEntries.Count - 1) / (targetCount - 1);

        for (int i = 0; i < targetCount; i++)
        {
            float exactIndex = i * stepSize;
            int lowerIndex = (int)Math.Floor(exactIndex);
            int upperIndex = Math.Min(lowerIndex + 1, sourceEntries.Count - 1);

            if (lowerIndex == upperIndex)
            {
                resampledEntries.Add(sourceEntries[lowerIndex]);
            }
            else
            {
                float alpha = exactIndex - lowerIndex;
                float interpolatedValue = sourceEntries[lowerIndex].Value * (1 - alpha) +
                                         sourceEntries[upperIndex].Value * alpha;

                resampledEntries.Add(new ScalarEntry(
                    sourceEntries[lowerIndex].Step,
                    interpolatedValue,
                    sourceEntries[lowerIndex].Timestamp));
            }
        }

        return new ScalarSeries(Name + "_resampled", resampledEntries);
    }

    private void UpdateStatistics()
    {
        lock (_lock)
        {
            if (_entries.Count == 0)
            {
                Min = null;
                Max = null;
                Average = 0;
                return;
            }

            Min = _entries[0].Value;
            Max = _entries[0].Value;
            float sum = 0;

            foreach (var entry in _entries)
            {
                sum += entry.Value;
                if (entry.Value < Min) Min = entry.Value;
                if (entry.Value > Max) Max = entry.Value;
            }

            Average = sum / _entries.Count;
        }
    }

    private void UpdateStatisticsIncremental(float newValue)
    {
        lock (_lock)
        {
            if (!Min.HasValue || newValue < Min) Min = newValue;
            if (!Max.HasValue || newValue > Max) Max = newValue;

            // Recalculate average (simplified approach)
            float sum = 0;
            foreach (var entry in _entries)
            {
                sum += entry.Value;
            }
            Average = sum / _entries.Count;
        }
    }
}
