using Xunit;
using ModelZoo.Benchmark;

namespace ModelZooTests.Benchmark;

/// <summary>
/// Unit tests for the BenchmarkHistory class.
/// </summary>
public class BenchmarkHistoryTests : IDisposable
{
    private readonly string _tempPath;
    private readonly BenchmarkHistory _history;

    public BenchmarkHistoryTests()
    {
        _tempPath = Path.Combine(Path.GetTempPath(), $"benchmark_test_{Guid.NewGuid()}");
        _history = new BenchmarkHistory(_tempPath);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempPath))
        {
            Directory.Delete(_tempPath, true);
        }
    }

    [Fact]
    public void SaveResult_WithValidResult_SavesToHistory()
    {
        // Arrange
        var result = CreateBenchmarkResult("model1");

        // Act
        _history.SaveResult(result);

        // Assert
        var retrieved = _history.GetLatest("model1");
        Assert.NotNull(retrieved);
        Assert.Equal("model1", retrieved.ModelName);
    }

    [Fact]
    public void SaveResult_WithEmptyModelName_ThrowsArgumentException()
    {
        // Arrange
        var result = new BenchmarkResult { ModelName = "" };

        // Act & Assert
        Assert.Throws<ArgumentException>(() => _history.SaveResult(result));
    }

    [Fact]
    public void SaveResult_WithNullResult_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => _history.SaveResult(null!));
    }

    [Fact]
    public void GetHistory_WithLimit_ReturnsCorrectNumberOfResults()
    {
        // Arrange
        for (int i = 0; i < 10; i++)
        {
            var result = CreateBenchmarkResult("model1");
            result.Timestamp = DateTime.UtcNow.AddMinutes(-i);
            _history.SaveResult(result);
        }

        // Act
        var history = _history.GetHistory("model1", 5);

        // Assert
        Assert.Equal(5, history.Count);
    }

    [Fact]
    public void GetHistory_WithNonExistentModel_ReturnsEmptyList()
    {
        // Act
        var history = _history.GetHistory("nonexistent");

        // Assert
        Assert.Empty(history);
    }

    [Fact]
    public void GetHistory_ResultsAreSortedByTimestamp_NewestFirst()
    {
        // Arrange
        var older = CreateBenchmarkResult("model1");
        older.Timestamp = DateTime.UtcNow.AddHours(-2);

        var newer = CreateBenchmarkResult("model1");
        newer.Timestamp = DateTime.UtcNow.AddHours(-1);

        _history.SaveResult(older);
        _history.SaveResult(newer);

        // Act
        var history = _history.GetHistory("model1");

        // Assert
        Assert.Equal(2, history.Count);
        Assert.True(history[0].Timestamp > history[1].Timestamp);
    }

    [Fact]
    public void GetLatest_WithResults_ReturnsMostRecent()
    {
        // Arrange
        var older = CreateBenchmarkResult("model1");
        older.Timestamp = DateTime.UtcNow.AddHours(-1);

        var newer = CreateBenchmarkResult("model1");
        newer.Timestamp = DateTime.UtcNow;

        _history.SaveResult(older);
        _history.SaveResult(newer);

        // Act
        var latest = _history.GetLatest("model1");

        // Assert
        Assert.NotNull(latest);
        Assert.Equal(newer.Timestamp, latest.Timestamp);
    }

    [Fact]
    public void GetLatest_WithNoResults_ReturnsNull()
    {
        // Act
        var latest = _history.GetLatest("nonexistent");

        // Assert
        Assert.Null(latest);
    }

    [Fact]
    public void CompareWithPrevious_WithPreviousResult_ReturnsComparison()
    {
        // Arrange
        var previous = CreateBenchmarkResult("model1");
        previous.Timestamp = DateTime.UtcNow.AddHours(-1);
        previous.Throughput = 100f;
        previous.AvgLatency = 10f;
        previous.Accuracy = 0.9f;
        previous.MemoryPeak = 1000000L;

        var current = CreateBenchmarkResult("model1");
        current.Timestamp = DateTime.UtcNow;
        current.Throughput = 110f;
        current.AvgLatency = 9f;
        current.Accuracy = 0.92f;
        current.MemoryPeak = 1100000L;

        _history.SaveResult(previous);
        _history.SaveResult(current);

        // Act
        var comparison = _history.CompareWithPrevious(current);

        // Assert
        Assert.NotNull(comparison);
        Assert.Equal(10f, comparison.ThroughputDiff);
        Assert.Equal(10f, comparison.ThroughputDiffPercent, 2);
        Assert.Equal(-1f, comparison.LatencyDiff);
        Assert.Equal(-10f, comparison.LatencyDiffPercent, 2);
        Assert.Equal(0.02f, comparison.AccuracyDiff, 3);
        Assert.Equal(100000L, comparison.MemoryPeakDiff);
    }

    [Fact]
    public void CompareWithPrevious_WithOnlyOneResult_ReturnsNull()
    {
        // Arrange
        var result = CreateBenchmarkResult("model1");
        _history.SaveResult(result);

        // Act
        var comparison = _history.CompareWithPrevious(result);

        // Assert
        Assert.Null(comparison);
    }

    [Fact]
    public void GetHistoryInRange_WithValidRange_ReturnsResultsInRange()
    {
        // Arrange
        var now = DateTime.UtcNow;
        var result1 = CreateBenchmarkResult("model1");
        result1.Timestamp = now.AddHours(-2);

        var result2 = CreateBenchmarkResult("model1");
        result2.Timestamp = now.AddHours(-1);

        var result3 = CreateBenchmarkResult("model1");
        result3.Timestamp = now;

        _history.SaveResult(result1);
        _history.SaveResult(result2);
        _history.SaveResult(result3);

        // Act
        var rangeResults = _history.GetHistoryInRange(
            "model1",
            now.AddHours(-1.5),
            now);

        // Assert
        Assert.Equal(2, rangeResults.Count);
    }

    [Fact]
    public void ClearHistory_WithExistingHistory_ClearsHistory()
    {
        // Arrange
        var result = CreateBenchmarkResult("model1");
        _history.SaveResult(result);

        // Act
        _history.ClearHistory("model1");
        var retrieved = _history.GetLatest("model1");

        // Assert
        Assert.Null(retrieved);
    }

    [Fact]
    public void ClearAll_WithExistingHistory_ClearsAllHistory()
    {
        // Arrange
        _history.SaveResult(CreateBenchmarkResult("model1"));
        _history.SaveResult(CreateBenchmarkResult("model2"));

        // Act
        _history.ClearAll();

        // Assert
        Assert.Equal(0, _history.TotalCount);
        Assert.Empty(_history.ModelNames);
    }

    [Fact]
    public void TotalCount_WithMultipleModels_ReturnsCorrectCount()
    {
        // Arrange
        _history.SaveResult(CreateBenchmarkResult("model1"));
        _history.SaveResult(CreateBenchmarkResult("model1"));
        _history.SaveResult(CreateBenchmarkResult("model2"));

        // Act
        var count = _history.TotalCount;

        // Assert
        Assert.Equal(3, count);
    }

    [Fact]
    public void ModelNames_WithMultipleModels_ReturnsAllModelNames()
    {
        // Arrange
        _history.SaveResult(CreateBenchmarkResult("model1"));
        _history.SaveResult(CreateBenchmarkResult("model2"));
        _history.SaveResult(CreateBenchmarkResult("model3"));

        // Act
        var names = _history.ModelNames.ToList();

        // Assert
        Assert.Equal(3, names.Count);
        Assert.Contains("model1", names);
        Assert.Contains("model2", names);
        Assert.Contains("model3", names);
    }

    [Fact]
    public void ExportToJson_WithHistory_CreatesValidJsonFile()
    {
        // Arrange
        _history.SaveResult(CreateBenchmarkResult("model1"));
        _history.SaveResult(CreateBenchmarkResult("model2"));
        var outputPath = Path.Combine(_tempPath, "export.json");

        // Act
        _history.ExportToJson(outputPath);

        // Assert
        Assert.True(File.Exists(outputPath));
        var content = File.ReadAllText(outputPath);
        Assert.Contains("model1", content);
        Assert.Contains("model2", content);
    }

    private static BenchmarkResult CreateBenchmarkResult(string modelName)
    {
        return new BenchmarkResult
        {
            ModelName = modelName,
            Dataset = "test_dataset",
            TotalSamples = 100,
            Throughput = 100.0f,
            AvgLatency = 10.0f,
            MinLatency = 5.0f,
            MaxLatency = 20.0f,
            P50Latency = 10.0f,
            P95Latency = 15.0f,
            P99Latency = 18.0f,
            MemoryPeak = 1000000L,
            MemoryAvg = 900000L,
            Accuracy = 0.9f,
            BenchmarkDuration = TimeSpan.FromSeconds(1),
            Timestamp = DateTime.UtcNow
        };
    }
}
