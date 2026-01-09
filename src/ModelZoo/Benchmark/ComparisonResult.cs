namespace ModelZoo.Benchmark;

/// <summary>
/// Results from comparing multiple models.
/// </summary>
public class ComparisonResult
{
    /// <summary>
    /// Gets or sets the benchmark results for each model.
    /// </summary>
    public Dictionary<string, BenchmarkResult> ModelResults { get; set; } = new Dictionary<string, BenchmarkResult>();

    /// <summary>
    /// Gets or sets the winning model based on the primary metric.
    /// </summary>
    public string? Winner { get; set; }

    /// <summary>
    /// Gets or sets rankings of models by each metric.
    /// Keys are metric names (e.g., "latency", "throughput").
    /// Values are arrays of model names sorted by the metric.
    /// </summary>
    public Dictionary<string, string[]> RankByMetric { get; set; } = new Dictionary<string, string[]>();

    /// <summary>
    /// Gets or sets whether the differences between models are statistically significant.
    /// Keys are model names (format: "model1_vs_model2").
    /// </summary>
    public Dictionary<string, bool> StatisticalSignificance { get; set; } = new Dictionary<string, bool>();

    /// <summary>
    /// Gets or sets the comparison configuration used.
    /// </summary>
    public ComparisonOptions? Options { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the comparison was run.
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets the total comparison duration.
    /// </summary>
    public TimeSpan TotalDuration => CalculateTotalDuration();

    /// <summary>
    /// Gets the number of models compared.
    /// </summary>
    public int ModelCount => ModelResults.Count;

    /// <summary>
    /// Calculates the total duration of all benchmarks.
    /// </summary>
    private TimeSpan CalculateTotalDuration()
    {
        return ModelResults.Values.Aggregate(
            TimeSpan.Zero,
            (sum, result) => sum + result.BenchmarkDuration);
    }

    /// <summary>
    /// Gets a summary of the comparison results.
    /// </summary>
    /// <returns>A human-readable summary.</returns>
    public string GetSummary()
    {
        var summary = $"Comparison of {ModelCount} models\n";
        summary += $"Winner: {Winner ?? "N/A"}\n\n";

        foreach (var metric in RankByMetric)
        {
            summary += $"{metric.Key} ranking: {string.Join(" > ", metric.Value)}\n";
        }

        return summary;
    }

    /// <summary>
    /// Gets the benchmark result for a specific model.
    /// </summary>
    /// <param name="modelName">The name of the model.</param>
    /// <returns>The benchmark result, or null if not found.</returns>
    public BenchmarkResult? GetModelResult(string modelName)
    {
        return ModelResults.TryGetValue(modelName, out var result) ? result : null;
    }

    /// <summary>
    /// Checks if the difference between two models is statistically significant.
    /// </summary>
    /// <param name="model1">The first model name.</param>
    /// <param name="model2">The second model name.</param>
    /// <returns>True if the difference is significant, false otherwise.</returns>
    public bool IsSignificant(string model1, string model2)
    {
        var key1 = $"{model1}_vs_{model2}";
        var key2 = $"{model2}_vs_{model1}";

        return StatisticalSignificance.TryGetValue(key1, out var result) ||
               StatisticalSignificance.TryGetValue(key2, out result) ?
               result : false;
    }
}
