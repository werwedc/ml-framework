using System.Text.Json;

namespace ModelZoo.Benchmark;

/// <summary>
/// Stores and retrieves historical benchmark results.
/// </summary>
public class BenchmarkHistory
{
    private readonly Dictionary<string, List<BenchmarkResult>> _history;
    private readonly string _storagePath;

    /// <summary>
    /// Initializes a new instance of the BenchmarkHistory class with in-memory storage.
    /// </summary>
    public BenchmarkHistory()
    {
        _history = new Dictionary<string, List<BenchmarkResult>>();
        _storagePath = string.Empty;
    }

    /// <summary>
    /// Initializes a new instance of the BenchmarkHistory class with persistent storage.
    /// </summary>
    /// <param name="storagePath">The directory path for storing benchmark history.</param>
    public BenchmarkHistory(string storagePath)
    {
        _history = new Dictionary<string, List<BenchmarkResult>>();
        _storagePath = storagePath ?? throw new ArgumentNullException(nameof(storagePath));

        // Create directory if it doesn't exist
        Directory.CreateDirectory(_storagePath);

        // Load existing history
        LoadHistory();
    }

    /// <summary>
    /// Gets the total number of benchmark results stored.
    /// </summary>
    public int TotalCount => _history.Values.Sum(list => list.Count);

    /// <summary>
    /// Gets the list of model names with stored benchmarks.
    /// </summary>
    public IEnumerable<string> ModelNames => _history.Keys;

    /// <summary>
    /// Saves a benchmark result to history.
    /// </summary>
    /// <param name="result">The benchmark result to save.</param>
    public void SaveResult(BenchmarkResult result)
    {
        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        if (string.IsNullOrWhiteSpace(result.ModelName))
        {
            throw new ArgumentException("Model name must not be empty.", nameof(result));
        }

        if (!_history.ContainsKey(result.ModelName))
        {
            _history[result.ModelName] = new List<BenchmarkResult>();
        }

        _history[result.ModelName].Add(result);

        // Sort by timestamp (newest first)
        _history[result.ModelName].Sort((a, b) => b.Timestamp.CompareTo(a.Timestamp));

        // Persist to disk if storage path is configured
        if (!string.IsNullOrEmpty(_storagePath))
        {
            SaveToDisk(result);
        }
    }

    /// <summary>
    /// Gets recent benchmarks for a model.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="limit">The maximum number of results to return.</param>
    /// <returns>A list of benchmark results, sorted by timestamp (newest first).</returns>
    public List<BenchmarkResult> GetHistory(string modelName, int limit = 10)
    {
        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Model name must not be empty.", nameof(modelName));
        }

        if (!_history.ContainsKey(modelName))
        {
            return new List<BenchmarkResult>();
        }

        return _history[modelName].Take(limit).ToList();
    }

    /// <summary>
    /// Gets the latest benchmark result for a model.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <returns>The latest benchmark result, or null if not found.</returns>
    public BenchmarkResult? GetLatest(string modelName)
    {
        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Model name must not be empty.", nameof(modelName));
        }

        if (!_history.ContainsKey(modelName) || _history[modelName].Count == 0)
        {
            return null;
        }

        return _history[modelName][0]; // First is newest due to sorting
    }

    /// <summary>
    /// Compares a benchmark result with the previous run of the same model.
    /// </summary>
    /// <param name="result">The benchmark result to compare.</param>
    /// <returns>A comparison result with differences, or null if no previous result exists.</returns>
    public BenchmarkComparison? CompareWithPrevious(BenchmarkResult result)
    {
        if (result == null)
        {
            throw new ArgumentNullException(nameof(result));
        }

        if (string.IsNullOrWhiteSpace(result.ModelName))
        {
            throw new ArgumentException("Model name must not be empty.", nameof(result));
        }

        if (!_history.ContainsKey(result.ModelName) || _history[result.ModelName].Count < 2)
        {
            return null;
        }

        // Get the previous result (index 1, since index 0 is the current result)
        var previous = _history[result.ModelName][1];

        return new BenchmarkComparison
        {
            Current = result,
            Previous = previous,
            ThroughputDiff = result.Throughput - previous.Throughput,
            ThroughputDiffPercent = previous.Throughput > 0
                ? ((result.Throughput - previous.Throughput) / previous.Throughput) * 100
                : 0,
            LatencyDiff = result.AvgLatency - previous.AvgLatency,
            LatencyDiffPercent = previous.AvgLatency > 0
                ? ((result.AvgLatency - previous.AvgLatency) / previous.AvgLatency) * 100
                : 0,
            AccuracyDiff = result.Accuracy - previous.Accuracy,
            AccuracyDiffPercent = previous.Accuracy > 0
                ? ((result.Accuracy - previous.Accuracy) / previous.Accuracy) * 100
                : 0,
            MemoryPeakDiff = result.MemoryPeak - previous.MemoryPeak,
            TimeSincePrevious = result.Timestamp - previous.Timestamp
        };
    }

    /// <summary>
    /// Gets benchmarks for a model within a time range.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <param name="startTime">The start of the time range.</param>
    /// <param name="endTime">The end of the time range.</param>
    /// <returns>A list of benchmark results within the time range.</returns>
    public List<BenchmarkResult> GetHistoryInRange(
        string modelName,
        DateTime startTime,
        DateTime endTime)
    {
        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Model name must not be empty.", nameof(modelName));
        }

        if (!_history.ContainsKey(modelName))
        {
            return new List<BenchmarkResult>();
        }

        return _history[modelName]
            .Where(r => r.Timestamp >= startTime && r.Timestamp <= endTime)
            .OrderBy(r => r.Timestamp)
            .ToList();
    }

    /// <summary>
    /// Clears all history for a model.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    public void ClearHistory(string modelName)
    {
        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Model name must not be empty.", nameof(modelName));
        }

        _history.Remove(modelName);

        // Remove from disk if storage path is configured
        if (!string.IsNullOrEmpty(_storagePath))
        {
            var filePath = GetStoragePath(modelName);
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }
        }
    }

    /// <summary>
    /// Clears all benchmark history.
    /// </summary>
    public void ClearAll()
    {
        _history.Clear();

        // Remove all files from storage if path is configured
        if (!string.IsNullOrEmpty(_storagePath) && Directory.Exists(_storagePath))
        {
            var files = Directory.GetFiles(_storagePath, "*.json");
            foreach (var file in files)
            {
                File.Delete(file);
            }
        }
    }

    /// <summary>
    /// Exports all benchmark history to a JSON file.
    /// </summary>
    /// <param name="filePath">The file path to export to.</param>
    public void ExportToJson(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path must not be empty.", nameof(filePath));
        }

        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        var json = JsonSerializer.Serialize(_history, options);
        File.WriteAllText(filePath, json);
    }

    /// <summary>
    /// Loads benchmark history from disk.
    /// </summary>
    private void LoadHistory()
    {
        if (string.IsNullOrEmpty(_storagePath) || !Directory.Exists(_storagePath))
        {
            return;
        }

        var files = Directory.GetFiles(_storagePath, "*.json");

        foreach (var file in files)
        {
            try
            {
                var json = File.ReadAllText(file);
                var options = new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                };

                var results = JsonSerializer.Deserialize<List<BenchmarkResult>>(json, options);
                if (results != null)
                {
                    var modelName = Path.GetFileNameWithoutExtension(file);
                    _history[modelName] = results;
                    _history[modelName].Sort((a, b) => b.Timestamp.CompareTo(a.Timestamp));
                }
            }
            catch (Exception ex)
            {
                // Log error but continue loading other files
                Console.WriteLine($"Warning: Failed to load history from {file}: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Saves a benchmark result to disk.
    /// </summary>
    private void SaveToDisk(BenchmarkResult result)
    {
        if (string.IsNullOrEmpty(_storagePath))
        {
            return;
        }

        var filePath = GetStoragePath(result.ModelName);
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        var json = JsonSerializer.Serialize(_history[result.ModelName], options);
        File.WriteAllText(filePath, json);
    }

    /// <summary>
    /// Gets the storage file path for a model.
    /// </summary>
    private string GetStoragePath(string modelName)
    {
        // Sanitize model name for use in file path
        var sanitizedName = string.Join("_", modelName.Split(Path.GetInvalidFileNameChars()));
        return Path.Combine(_storagePath, $"{sanitizedName}.json");
    }
}

/// <summary>
/// Comparison between two benchmark results.
/// </summary>
public class BenchmarkComparison
{
    /// <summary>
    /// Gets or sets the current benchmark result.
    /// </summary>
    public BenchmarkResult? Current { get; set; }

    /// <summary>
    /// Gets or sets the previous benchmark result.
    /// </summary>
    public BenchmarkResult? Previous { get; set; }

    /// <summary>
    /// Gets or sets the difference in throughput (samples per second).
    /// </summary>
    public float ThroughputDiff { get; set; }

    /// <summary>
    /// Gets or sets the percentage difference in throughput.
    /// </summary>
    public float ThroughputDiffPercent { get; set; }

    /// <summary>
    /// Gets or sets the difference in average latency (milliseconds).
    /// </summary>
    public float LatencyDiff { get; set; }

    /// <summary>
    /// Gets or sets the percentage difference in latency.
    /// </summary>
    public float LatencyDiffPercent { get; set; }

    /// <summary>
    /// Gets or sets the difference in accuracy.
    /// </summary>
    public float AccuracyDiff { get; set; }

    /// <summary>
    /// Gets or sets the percentage difference in accuracy.
    /// </summary>
    public float AccuracyDiffPercent { get; set; }

    /// <summary>
    /// Gets or sets the difference in peak memory usage (bytes).
    /// </summary>
    public long MemoryPeakDiff { get; set; }

    /// <summary>
    /// Gets or sets the time elapsed between the two benchmark runs.
    /// </summary>
    public TimeSpan TimeSincePrevious { get; set; }

    /// <summary>
    /// Returns a summary of the comparison.
    /// </summary>
    /// <returns>A human-readable summary.</returns>
    public string GetSummary()
    {
        var summary = $"Benchmark Comparison:\n";
        summary += $"  Time since previous: {TimeSincePrevious.TotalDays:F1} days\n";
        summary += $"  Throughput: {(ThroughputDiffPercent >= 0 ? "+" : "")}{ThroughputDiffPercent:F2}% " +
                   $"({ThroughputDiff:F2} samples/s)\n";
        summary += $"  Latency: {(LatencyDiffPercent >= 0 ? "+" : "")}{LatencyDiffPercent:F2}% " +
                   $"({LatencyDiff:F2}ms)\n";
        summary += $"  Accuracy: {(AccuracyDiffPercent >= 0 ? "+" : "")}{AccuracyDiffPercent:F2}% " +
                   $"({AccuracyDiff:F4})\n";
        summary += $"  Memory Peak: {(MemoryPeakDiff >= 0 ? "+" : "")}{MemoryPeakDiff:N0} bytes";

        return summary;
    }
}
