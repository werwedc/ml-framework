using Xunit;
using ModelZoo.Benchmark;
using System.Text.Json;

namespace ModelZooTests.Benchmark;

/// <summary>
/// Unit tests for the BenchmarkReporter class.
/// </summary>
public class BenchmarkReporterTests : IDisposable
{
    private readonly string _tempPath;
    private readonly BenchmarkReporter _reporter;

    public BenchmarkReporterTests()
    {
        _tempPath = Path.Combine(Path.GetTempPath(), $"benchmark_report_test_{Guid.NewGuid()}");
        Directory.CreateDirectory(_tempPath);
        _reporter = new BenchmarkReporter();
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempPath))
        {
            Directory.Delete(_tempPath, true);
        }
    }

    [Fact]
    public void GenerateReport_WithTextFormat_CreatesTextFile()
    {
        // Arrange
        var result = CreateBenchmarkResult();
        var outputPath = Path.Combine(_tempPath, "report.txt");

        // Act
        _reporter.GenerateReport(result, outputPath, ReportFormat.Text);

        // Assert
        Assert.True(File.Exists(outputPath));
        var content = File.ReadAllText(outputPath);
        Assert.Contains("MODEL BENCHMARK REPORT", content);
        Assert.Contains(result.ModelName, content);
        Assert.Contains(result.Dataset, content);
    }

    [Fact]
    public void GenerateReport_WithMarkdownFormat_CreatesMarkdownFile()
    {
        // Arrange
        var result = CreateBenchmarkResult();
        var outputPath = Path.Combine(_tempPath, "report.md");

        // Act
        _reporter.GenerateReport(result, outputPath, ReportFormat.Markdown);

        // Assert
        Assert.True(File.Exists(outputPath));
        var content = File.ReadAllText(outputPath);
        Assert.Contains("# Model Benchmark Report", content);
        Assert.Contains(result.ModelName, content);
        Assert.Contains("|", content); // Markdown table format
    }

    [Fact]
    public void GenerateReport_WithJsonFormat_CreatesValidJsonFile()
    {
        // Arrange
        var result = CreateBenchmarkResult();
        var outputPath = Path.Combine(_tempPath, "report.json");

        // Act
        _reporter.GenerateReport(result, outputPath, ReportFormat.Json);

        // Assert
        Assert.True(File.Exists(outputPath));
        var content = File.ReadAllText(outputPath);

        // Verify it's valid JSON
        var jsonDoc = JsonDocument.Parse(content);
        Assert.NotNull(jsonDoc);
        Assert.Equal(result.ModelName, jsonDoc.RootElement.GetProperty("modelName").GetString());
    }

    [Fact]
    public void GenerateReport_WithCsvFormat_CreatesCsvFile()
    {
        // Arrange
        var result = CreateBenchmarkResult();
        var outputPath = Path.Combine(_tempPath, "report.csv");

        // Act
        _reporter.GenerateReport(result, outputPath, ReportFormat.Csv);

        // Assert
        Assert.True(File.Exists(outputPath));
        var content = File.ReadAllText(outputPath);
        Assert.Contains("metric,value", content);
        Assert.Contains("model_name", content);
    }

    [Fact]
    public void GenerateComparisonReport_WithTextFormat_CreatesTextFile()
    {
        // Arrange
        var result = CreateComparisonResult();
        var outputPath = Path.Combine(_tempPath, "comparison.txt");

        // Act
        _reporter.GenerateComparisonReport(result, outputPath, ReportFormat.Text);

        // Assert
        Assert.True(File.Exists(outputPath));
        var content = File.ReadAllText(outputPath);
        Assert.Contains("MODEL COMPARISON REPORT", content);
        Assert.Contains(result.Winner, content!);
    }

    [Fact]
    public void GenerateComparisonReport_WithMarkdownFormat_CreatesMarkdownFile()
    {
        // Arrange
        var result = CreateComparisonResult();
        var outputPath = Path.Combine(_tempPath, "comparison.md");

        // Act
        _reporter.GenerateComparisonReport(result, outputPath, ReportFormat.Markdown);

        // Assert
        Assert.True(File.Exists(outputPath));
        var content = File.ReadAllText(outputPath);
        Assert.Contains("# Model Comparison Report", content);
        Assert.Contains("|", content); // Markdown table format
    }

    [Fact]
    public void GenerateComparisonReport_WithJsonFormat_CreatesValidJsonFile()
    {
        // Arrange
        var result = CreateComparisonResult();
        var outputPath = Path.Combine(_tempPath, "comparison.json");

        // Act
        _reporter.GenerateComparisonReport(result, outputPath, ReportFormat.Json);

        // Assert
        Assert.True(File.Exists(outputPath));
        var content = File.ReadAllText(outputPath);

        // Verify it's valid JSON
        var jsonDoc = JsonDocument.Parse(content);
        Assert.NotNull(jsonDoc);
        Assert.Equal(result.Winner, jsonDoc.RootElement.GetProperty("winner").GetString());
    }

    [Fact]
    public void GenerateComparisonReport_WithCsvFormat_CreatesCsvFile()
    {
        // Arrange
        var result = CreateComparisonResult();
        var outputPath = Path.Combine(_tempPath, "comparison.csv");

        // Act
        _reporter.GenerateComparisonReport(result, outputPath, ReportFormat.Csv);

        // Assert
        Assert.True(File.Exists(outputPath));
        var content = File.ReadAllText(outputPath);
        Assert.Contains("model,throughput", content);
    }

    [Fact]
    public void PrintResult_WithValidResult_DoesNotThrow()
    {
        // Arrange
        var result = CreateBenchmarkResult();

        // Act & Assert - Should not throw
        _reporter.PrintResult(result);
    }

    [Fact]
    public void PrintComparison_WithValidResult_DoesNotThrow()
    {
        // Arrange
        var result = CreateComparisonResult();

        // Act & Assert - Should not throw
        _reporter.PrintComparison(result);
    }

    [Fact]
    public void GenerateReport_WithNullResult_ThrowsArgumentNullException()
    {
        // Arrange
        var outputPath = Path.Combine(_tempPath, "report.txt");

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            _reporter.GenerateReport(null!, outputPath));
    }

    [Fact]
    public void GenerateReport_WithEmptyPath_ThrowsArgumentException()
    {
        // Arrange
        var result = CreateBenchmarkResult();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            _reporter.GenerateReport(result, ""));
    }

    [Fact]
    public void GenerateComparisonReport_WithNullResult_ThrowsArgumentNullException()
    {
        // Arrange
        var outputPath = Path.Combine(_tempPath, "comparison.txt");

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            _reporter.GenerateComparisonReport(null!, outputPath));
    }

    [Fact]
    public void GenerateReport_ContainsExpectedMetrics()
    {
        // Arrange
        var result = CreateBenchmarkResult();
        var outputPath = Path.Combine(_tempPath, "report.txt");

        // Act
        _reporter.GenerateReport(result, outputPath, ReportFormat.Text);
        var content = File.ReadAllText(outputPath);

        // Assert
        Assert.Contains("Throughput", content);
        Assert.Contains("Latency", content);
        Assert.Contains("Accuracy", content);
        Assert.Contains("Memory", content);
    }

    [Fact]
    public void GenerateComparisonReport_ContainsRankings()
    {
        // Arrange
        var result = CreateComparisonResult();
        var outputPath = Path.Combine(_tempPath, "comparison.txt");

        // Act
        _reporter.GenerateComparisonReport(result, outputPath, ReportFormat.Text);
        var content = File.ReadAllText(outputPath);

        // Assert
        Assert.Contains("RANKINGS BY METRIC", content);
        Assert.Contains("Winner", content);
    }

    private static BenchmarkResult CreateBenchmarkResult()
    {
        return new BenchmarkResult
        {
            ModelName = "test_model",
            Dataset = "test_dataset",
            TotalSamples = 1000,
            Throughput = 100.5f,
            AvgLatency = 9.95f,
            MinLatency = 5.0f,
            MaxLatency = 20.0f,
            P50Latency = 10.0f,
            P95Latency = 15.0f,
            P99Latency = 18.0f,
            MemoryPeak = 1073741824L, // 1 GB
            MemoryAvg = 1048576000L,   // ~1 GB
            Accuracy = 0.95f,
            BenchmarkDuration = TimeSpan.FromSeconds(10),
            Timestamp = DateTime.UtcNow
        };
    }

    private static ComparisonResult CreateComparisonResult()
    {
        var result = new ComparisonResult
        {
            Winner = "model1",
            Timestamp = DateTime.UtcNow
        };

        // Add model results
        result.ModelResults["model1"] = new BenchmarkResult
        {
            ModelName = "model1",
            Dataset = "test_dataset",
            TotalSamples = 1000,
            Throughput = 120.0f,
            AvgLatency = 8.33f,
            Accuracy = 0.96f,
            MemoryPeak = 1073741824L,
            BenchmarkDuration = TimeSpan.FromSeconds(10),
            Timestamp = DateTime.UtcNow
        };

        result.ModelResults["model2"] = new BenchmarkResult
        {
            ModelName = "model2",
            Dataset = "test_dataset",
            TotalSamples = 1000,
            Throughput = 100.0f,
            AvgLatency = 10.0f,
            Accuracy = 0.94f,
            MemoryPeak = 1073741824L,
            BenchmarkDuration = TimeSpan.FromSeconds(10),
            Timestamp = DateTime.UtcNow
        };

        result.ModelResults["model3"] = new BenchmarkResult
        {
            ModelName = "model3",
            Dataset = "test_dataset",
            TotalSamples = 1000,
            Throughput = 80.0f,
            AvgLatency = 12.5f,
            Accuracy = 0.92f,
            MemoryPeak = 1073741824L,
            BenchmarkDuration = TimeSpan.FromSeconds(10),
            Timestamp = DateTime.UtcNow
        };

        // Add rankings
        result.RankByMetric["throughput"] = new[] { "model1", "model2", "model3" };
        result.RankByMetric["latency"] = new[] { "model1", "model2", "model3" };
        result.RankByMetric["accuracy"] = new[] { "model1", "model2", "model3" };

        // Add statistical significance
        result.StatisticalSignificance["model1_vs_model2"] = true;
        result.StatisticalSignificance["model1_vs_model3"] = true;
        result.StatisticalSignificance["model2_vs_model3"] = false;

        return result;
    }
}
