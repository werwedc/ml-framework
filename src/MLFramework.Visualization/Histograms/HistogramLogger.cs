using MachineLearning.Visualization.Histograms;

namespace MLFramework.Visualization.Histograms;

/// <summary>
/// Implementation of IHistogramLogger for tracking tensor distributions
/// </summary>
public class HistogramLogger : IHistogramLogger
{
    private readonly Dictionary<string, List<HistogramData>> _histograms = new();
    private readonly Dictionary<string, List<TensorDistribution>> _distributions = new();
    private readonly Dictionary<string, long> _stepCounters = new();
    private readonly object _lock = new();

    /// <summary>
    /// Gets the default binning configuration
    /// </summary>
    public HistogramBinConfig DefaultBinConfig { get; set; } = new HistogramBinConfig();

    /// <summary>
    /// Gets or sets whether to automatically detect outliers
    /// </summary>
    public bool AutoDetectOutliers { get; set; } = true;

    /// <summary>
    /// Gets or sets the outlier threshold in standard deviations
    /// </summary>
    public float OutlierThresholdStd { get; set; } = 3.0f;

    /// <summary>
    /// Creates a new HistogramLogger
    /// </summary>
    public HistogramLogger()
    {
    }

    /// <summary>
    /// Logs a histogram synchronously with default configuration
    /// </summary>
    public void LogHistogram(string name, float[] values, long step = -1)
    {
        LogHistogram(name, values, DefaultBinConfig, step);
    }

    /// <summary>
    /// Logs a histogram synchronously with custom configuration
    /// </summary>
    public void LogHistogram(string name, float[] values, HistogramBinConfig config, long step = -1)
    {
        if (string.IsNullOrEmpty(name))
        {
            throw new ArgumentException("Name cannot be null or empty", nameof(name));
        }

        if (values == null || values.Length == 0)
        {
            throw new ArgumentException("Values array cannot be null or empty", nameof(values));
        }

        if (config == null)
        {
            throw new ArgumentNullException(nameof(config));
        }

        lock (_lock)
        {
            // Auto-increment step if needed
            if (step == -1)
            {
                if (!_stepCounters.ContainsKey(name))
                {
                    _stepCounters[name] = 0;
                }
                step = _stepCounters[name]++;
            }

            // Create histogram
            var histogram = HistogramData.Create(name, values, config, step);

            // Store histogram
            if (!_histograms.ContainsKey(name))
            {
                _histograms[name] = new List<HistogramData>();
            }

            _histograms[name].Add(histogram);

            // Publish event (in a real implementation, this would use an event publisher)
            var histogramEvent = new HistogramEvent(histogram);
            // TODO: Publish to event system when available
        }
    }

    /// <summary>
    /// Logs a histogram asynchronously with default configuration
    /// </summary>
    public Task LogHistogramAsync(string name, float[] values, long step = -1)
    {
        return Task.Run(() => LogHistogram(name, values, step));
    }

    /// <summary>
    /// Logs a histogram asynchronously with custom configuration
    /// </summary>
    public Task LogHistogramAsync(string name, float[] values, HistogramBinConfig config, long step = -1)
    {
        return Task.Run(() => LogHistogram(name, values, config, step));
    }

    /// <summary>
    /// Logs a tensor distribution synchronously
    /// </summary>
    public void LogDistribution(string name, float[] values, long step = -1)
    {
        if (string.IsNullOrEmpty(name))
        {
            throw new ArgumentException("Name cannot be null or empty", nameof(name));
        }

        if (values == null || values.Length == 0)
        {
            throw new ArgumentException("Values array cannot be null or empty", nameof(values));
        }

        lock (_lock)
        {
            // Auto-increment step if needed
            if (step == -1)
            {
                if (!_stepCounters.ContainsKey(name))
                {
                    _stepCounters[name] = 0;
                }
                step = _stepCounters[name]++;
            }

            // Create distribution
            var distribution = TensorDistribution.Create(
                name,
                values,
                DefaultBinConfig,
                step,
                AutoDetectOutliers,
                OutlierThresholdStd);

            // Store distribution
            if (!_distributions.ContainsKey(name))
            {
                _distributions[name] = new List<TensorDistribution>();
            }

            _distributions[name].Add(distribution);

            // Publish event (in a real implementation, this would use an event publisher)
            var histogramEvent = new HistogramEvent(distribution);
            // TODO: Publish to event system when available
        }
    }

    /// <summary>
    /// Logs a tensor distribution asynchronously
    /// </summary>
    public Task LogDistributionAsync(string name, float[] values, long step = -1)
    {
        return Task.Run(() => LogDistribution(name, values, step));
    }

    /// <summary>
    /// Retrieves a histogram by name and step
    /// </summary>
    public HistogramData? GetHistogram(string name, long step)
    {
        if (string.IsNullOrEmpty(name))
        {
            throw new ArgumentException("Name cannot be null or empty", nameof(name));
        }

        lock (_lock)
        {
            if (!_histograms.ContainsKey(name))
            {
                return null;
            }

            return _histograms[name].FirstOrDefault(h => h.Step == step);
        }
    }

    /// <summary>
    /// Retrieves a distribution by name and step
    /// </summary>
    public TensorDistribution? GetDistribution(string name, long step)
    {
        if (string.IsNullOrEmpty(name))
        {
            throw new ArgumentException("Name cannot be null or empty", nameof(name));
        }

        lock (_lock)
        {
            if (!_distributions.ContainsKey(name))
            {
                return null;
            }

            return _distributions[name].FirstOrDefault(d => d.Step == step);
        }
    }

    /// <summary>
    /// Retrieves all histograms for a given name over time
    /// </summary>
    public IEnumerable<HistogramData> GetHistogramsOverTime(string name)
    {
        if (string.IsNullOrEmpty(name))
        {
            throw new ArgumentException("Name cannot be null or empty", nameof(name));
        }

        lock (_lock)
        {
            if (!_histograms.ContainsKey(name))
            {
                return Enumerable.Empty<HistogramData>();
            }

            return _histograms[name].OrderBy(h => h.Step).ToArray();
        }
    }

    /// <summary>
    /// Retrieves all distributions for a given name over time
    /// </summary>
    public IEnumerable<TensorDistribution> GetDistributionsOverTime(string name)
    {
        if (string.IsNullOrEmpty(name))
        {
            throw new ArgumentException("Name cannot be null or empty", nameof(name));
        }

        lock (_lock)
        {
            if (!_distributions.ContainsKey(name))
            {
                return Enumerable.Empty<TensorDistribution>();
            }

            return _distributions[name].OrderBy(d => d.Step).ToArray();
        }
    }

    /// <summary>
    /// Gets all histogram names
    /// </summary>
    public IEnumerable<string> GetHistogramNames()
    {
        lock (_lock)
        {
            return _histograms.Keys.ToArray();
        }
    }

    /// <summary>
    /// Gets all distribution names
    /// </summary>
    public IEnumerable<string> GetDistributionNames()
    {
        lock (_lock)
        {
            return _distributions.Keys.ToArray();
        }
    }

    /// <summary>
    /// Clears all logged data
    /// </summary>
    public void Clear()
    {
        lock (_lock)
        {
            _histograms.Clear();
            _distributions.Clear();
            _stepCounters.Clear();
        }
    }

    /// <summary>
    /// Clears data for a specific name
    /// </summary>
    public void Clear(string name)
    {
        if (string.IsNullOrEmpty(name))
        {
            throw new ArgumentException("Name cannot be null or empty", nameof(name));
        }

        lock (_lock)
        {
            if (_histograms.ContainsKey(name))
            {
                _histograms[name].Clear();
            }

            if (_distributions.ContainsKey(name))
            {
                _distributions[name].Clear();
            }

            if (_stepCounters.ContainsKey(name))
            {
                _stepCounters.Remove(name);
            }
        }
    }
}
