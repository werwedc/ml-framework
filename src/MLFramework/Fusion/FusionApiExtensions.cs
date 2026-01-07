namespace MLFramework.Fusion;

/// <summary>
/// Extension methods and helper functions for fusion API
/// </summary>
public static class FusionApiExtensions
{
    /// <summary>
    /// Applies fusion to a computational graph
    /// </summary>
    public static ComputationalGraph ApplyFusion(
        this ComputationalGraph graph,
        IFusionEngine engine,
        FusionOptions? options = null)
    {
        var fusionOptions = options ?? new FusionOptions
        {
            EnableFusion = GraphOptions.EnableFusion,
            MaxFusionOps = GraphOptions.MaxFusionOps,
            MinBenefitScore = GraphOptions.MinBenefitScore,
            Aggressiveness = GraphOptions.Aggressiveness
        };

        var result = engine.ApplyFusion(graph, fusionOptions);
        return result.FusedGraph;
    }

    /// <summary>
    /// Gets all fused operations in the graph
    /// </summary>
    public static IReadOnlyList<FusedOperation> GetFusedOperations(this ComputationalGraph graph)
    {
        return graph.Operations.OfType<FusedOperation>().ToList();
    }

    /// <summary>
    /// Gets fusion statistics for a graph
    /// </summary>
    public static FusionGraphStatistics GetFusionStatistics(this ComputationalGraph graph)
    {
        var fusedOps = graph.GetFusedOperations();
        var totalOps = graph.Operations.Count;
        var fusedOpsCount = fusedOps.Sum(op => op.ConstituentOperations.Count);
        var rejectedOps = totalOps - fusedOps.Count;

        return new FusionGraphStatistics
        {
            TotalOperations = totalOps,
            FusedOperations = fusedOpsCount,
            FusedGroups = fusedOps.Count,
            RejectedOperations = rejectedOps,
            EstimatedSpeedup = EstimateSpeedup(fusedOps)
        };
    }

    /// <summary>
    /// Estimates the speedup from fusion (simple heuristic)
    /// </summary>
    private static double EstimateSpeedup(IReadOnlyList<FusedOperation> fusedOps)
    {
        // Simple heuristic: 1.3x speedup per fused operation group
        return fusedOps.Count * 1.3;
    }
}

/// <summary>
/// Statistics for a computational graph's fusion status
/// </summary>
public record FusionGraphStatistics
{
    public required int TotalOperations { get; init; }
    public required int FusedOperations { get; init; }
    public required int FusedGroups { get; init; }
    public required int RejectedOperations { get; init; }
    public required double EstimatedSpeedup { get; init; }
}
