using System;
using MLFramework.Modules;

namespace MLFramework.Conversion;

/// <summary>
/// Provides static methods to estimate memory usage for tensor parallelism.
/// </summary>
public static class TPMemoryEstimator
{
    /// <summary>
    /// Estimate memory usage for model with and without TP.
    /// </summary>
    /// <param name="model">The model to analyze.</param>
    /// <param name="worldSize">The world size for tensor parallelism.</param>
    /// <returns>A memory estimate containing the analysis results.</returns>
    public static MemoryEstimate EstimateMemory(IModule model, int worldSize)
    {
        var analysis = TPModelAnalyzer.Analyze(model, worldSize);
        var estimate = new MemoryEstimate();

        // Base memory (without TP)
        estimate.BaseMemoryMB = analysis.TotalMemoryBytes / 1024 / 1024;

        // Memory with TP (each rank stores only a fraction)
        estimate.TPMemoryPerRankMB = (long)(
            analysis.TotalMemoryBytes - analysis.ParallelizableMemoryBytes +
            analysis.ParallelizableMemoryBytes / worldSize
        ) / 1024 / 1024;

        // Communication overhead (temporary buffers)
        estimate.CommunicationOverheadMB = (long)(estimate.TPMemoryPerRankMB * 0.1); // 10% overhead

        // Total memory per rank
        estimate.TotalMemoryPerRankMB = estimate.TPMemoryPerRankMB + estimate.CommunicationOverheadMB;

        // Memory savings
        estimate.MemorySavingsPercentage = (
            1 - (double)estimate.TotalMemoryPerRankMB / estimate.BaseMemoryMB
        ) * 100;

        return estimate;
    }
}

/// <summary>
/// Contains memory estimation results for tensor parallelism.
/// </summary>
public class MemoryEstimate
{
    /// <summary>
    /// Gets or sets the base memory usage in MB without TP.
    /// </summary>
    public long BaseMemoryMB { get; set; }

    /// <summary>
    /// Gets or sets the memory usage in MB per rank with TP (excluding overhead).
    /// </summary>
    public long TPMemoryPerRankMB { get; set; }

    /// <summary>
    /// Gets or sets the communication overhead in MB.
    /// </summary>
    public long CommunicationOverheadMB { get; set; }

    /// <summary>
    /// Gets or sets the total memory per rank in MB including overhead.
    /// </summary>
    public long TotalMemoryPerRankMB { get; set; }

    /// <summary>
    /// Gets or sets the percentage of memory saved by using TP.
    /// </summary>
    public double MemorySavingsPercentage { get; set; }

    /// <summary>
    /// Generates a human-readable summary of the memory estimate.
    /// </summary>
    public string GenerateSummary()
    {
        return $"Base memory: {BaseMemoryMB} MB\n" +
               $"TP memory per rank: {TotalMemoryPerRankMB} MB\n" +
               $"Savings: {MemorySavingsPercentage:F1}%\n" +
               $"Communication overhead: {CommunicationOverheadMB} MB";
    }
}
